#!/usr/bin/env python3
"""Толян — голосовой собеседник для Discord через виртуальные аудиокабели.

Discord → Bot_Ear → VAD → STT → Ollama → Piper/edge-tts → Bot_Mouth → VirtMic → Discord
"""
import argparse
import asyncio
import collections
import json
import queue
import random
import re
import subprocess
import sys
import threading
import time

import edge_tts
import numpy as np
import requests
import speech_recognition as sr
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import audio
import config

try:
    import webrtcvad
except ImportError:  # без него работаем только по порогу громкости
    webrtcvad = None

console = Console(highlight=False)
FRAME_BYTES = config.SAMPLE_RATE * config.FRAME_MS // 1000 * 2  # s16le моно


def rms(pcm: bytes) -> float:
    if not pcm:
        return 0.0
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(samples * samples)))


def record_cmd(source: str) -> list[str]:
    return ["parec", f"--device={source}", "--format=s16le", f"--rate={config.SAMPLE_RATE}",
            "--channels=1", "--raw", "--latency-msec=30", "--client-name=Tolyan"]


# ───────────────────────────── Уши ─────────────────────────────

class Ear:
    """Слушает Bot_Ear.monitor и режет поток на фразы по паузам."""

    def __init__(self, muted: threading.Event, on_voice=lambda: None):
        self.muted = muted
        self.on_voice = on_voice
        self.vad = webrtcvad.Vad(config.VAD_MODE) if webrtcvad else None
        self.proc = None
        self.stopped = False
        self.is_speaking = False

    def _is_speech(self, frame: bytes) -> bool:
        if rms(frame) < config.RMS_THRESHOLD:
            return False
        return self.vad.is_speech(frame, config.SAMPLE_RATE) if self.vad else True

    def utterances(self):
        """Генератор: отдаёт сырые PCM-фразы (s16le, 16 кГц, моно)."""
        self.proc = subprocess.Popen(record_cmd(f"{config.EAR_SINK}.monitor"),
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        pre_roll = collections.deque(maxlen=10)   # ~300 мс до начала речи
        recent = collections.deque(maxlen=8)
        frames, voiced, silence = [], 0, 0
        max_frames = config.MAX_UTTERANCE_SEC * 1000 // config.FRAME_MS
        while not self.stopped:
            frame = self.proc.stdout.read(FRAME_BYTES)
            if len(frame) < FRAME_BYTES:
                break
            if self.muted.is_set():  # Толян говорит — уши на паузе
                pre_roll.clear(); recent.clear()
                frames, voiced, silence = [], 0, 0
                self.is_speaking = False
                continue
            speech = self._is_speech(frame)
            if not frames:
                pre_roll.append(frame)
                recent.append(speech)
                if sum(recent) >= 6:
                    frames = list(pre_roll)
                    voiced, silence = sum(recent), 0
                    self.is_speaking = True
                    self.on_voice()
                continue
            frames.append(frame)
            voiced += speech
            silence = 0 if speech else silence + 1
            if silence * config.FRAME_MS >= config.SILENCE_MS or len(frames) >= max_frames:
                self.is_speaking = False
                if voiced * config.FRAME_MS >= config.MIN_SPEECH_MS:
                    yield b"".join(frames)
                pre_roll.clear(); recent.clear()
                frames, voiced, silence = [], 0, 0

    def stop(self):
        self.stopped = True
        self.is_speaking = False
        if self.proc:
            self.proc.terminate()


# ───────────────────────────── STT ─────────────────────────────

_TYPO_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in config.NAME_TYPOS) + r")([а-яё]*)\b",
    re.IGNORECASE,
) if getattr(config, "NAME_TYPOS", ()) else None


def clean_stt_text(text: str) -> str:
    """Фильтрует фантомные субтитры Whisper и чинит частые ослышки имени «Толян»."""
    text = text.strip()
    if not text:
        return ""
    low = text.lower().replace("ё", "е")
    if any(h in low for h in getattr(config, "STT_HALLUCINATIONS", ())):
        return ""
    if _TYPO_RE:
        text = _TYPO_RE.sub(r"Толян\2", text)
    return text


class GoogleSTT:
    name = "Google Web Speech"

    def __init__(self):
        self.recognizer = sr.Recognizer()

    def transcribe(self, pcm: bytes) -> str:
        data = sr.AudioData(pcm, config.SAMPLE_RATE, 2)
        try:
            raw = self.recognizer.recognize_google(data, language=config.STT_LANGUAGE)
            return clean_stt_text(raw)
        except sr.UnknownValueError:
            return ""


class WhisperSTT:
    def __init__(self):
        from faster_whisper import WhisperModel
        self.name = f"faster-whisper {config.WHISPER_MODEL} (CPU, int8)"
        self.model = WhisperModel(
            config.WHISPER_MODEL, device="cpu", compute_type="int8", cpu_threads=4
        )

    def transcribe(self, pcm: bytes) -> str:
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = self.model.transcribe(
            samples,
            language="ru",
            beam_size=1,
            condition_on_previous_text=False,
            initial_prompt=getattr(config, "WHISPER_PROMPT", None),
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 400},
        )
        raw = " ".join(seg.text.strip() for seg in segments).strip()
        return clean_stt_text(raw)


def make_stt(engine: str):
    if engine == "whisper":
        return WhisperSTT()
    if engine == "google":
        return GoogleSTT()
    raise SystemExit(f"Неизвестный STT-движок: {engine} (нужен google или whisper)")


# ───────────────────────────── Мозги ─────────────────────────────

SENTENCE_END = re.compile(r"[.!?…]+[»\")]*(?=\s)")
MIN_CHUNK_CHARS = 28  # короткие выкрики («Ну да!») клеим к основной мысли, не тратя лимит предложений


def retrieve_lore(text: str, max_items: int = 2) -> str:
    """Подтягивает подходящие локальные мемы компании по ключевым словам из реплики."""
    lore_book = getattr(config, "LORE_BOOK", ())
    if not lore_book or not text:
        return ""
    norm = text.lower().replace("ё", "е")
    scored = []
    for entry in lore_book:
        hits = sum(1 for kw in entry.get("keywords", ()) if kw in norm)
        if hits > 0:
            scored.append((hits, entry["fact"]))
    scored.sort(key=lambda x: x[0], reverse=True)
    return " ".join(fact for _, fact in scored[:max_items])


class Brain:
    def __init__(self):
        self.history: list[dict] = []
        self.overheard: collections.deque[str] = collections.deque(
            maxlen=getattr(config, "OVERHEARD_MAX", 5)
        )
        self.memories: list[str] = self._load_memories()

    def _load_memories(self) -> list[str]:
        import os
        path = getattr(config, "MEMORY_FILE", "")
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return [str(x) for x in data[-getattr(config, "MEMORY_MAX_ITEMS", 25):]]
            except Exception:
                pass
        return []

    def remember(self, raw_text: str) -> str:
        """Сохраняет новый факт/мем в долговременную память memory.json."""
        fact = re.sub(
            r"(?i)^.*?\b(запомни|запиши на корочку|запиши в память)\b[\s,:—-]*",
            "",
            raw_text,
        ).strip()
        if not fact:
            fact = raw_text.strip()
        if fact and fact not in self.memories:
            self.memories.append(fact)
            max_items = getattr(config, "MEMORY_MAX_ITEMS", 25)
            if len(self.memories) > max_items:
                self.memories = self.memories[-max_items:]
            path = getattr(config, "MEMORY_FILE", "")
            if path:
                try:
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(self.memories, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass
        return fact

    def overhear(self, text: str) -> None:
        """Запоминает фоновые реплики пацанов, чтобы быть в курсе разговора, когда позовут."""
        if text:
            self.overheard.append(text)

    def reset(self) -> None:
        """Очищает память диалога (команда «Толян, забудь»)."""
        self.history.clear()
        self.overheard.clear()

    def _base_messages(self) -> list[dict]:
        sys_prompt = config.SYSTEM_PROMPT
        if self.memories:
            mem_block = "\n".join(f"- {m}" for m in self.memories)
            sys_prompt = f"{sys_prompt}\n\nСвежие события и факты, которые пацаны просили запомнить:\n{mem_block}"
        return [
            {"role": "system", "content": sys_prompt},
            *getattr(config, "FEW_SHOT_MESSAGES", []),
        ]

    def _request(self, messages: list[dict], stream: bool, num_predict: int):
        resp = requests.post(f"{config.OLLAMA_URL}/api/chat", timeout=180, stream=stream, json={
            "model": config.OLLAMA_MODEL,
            "messages": messages,
            "stream": stream,
            "keep_alive": -1,  # модель всегда висит в VRAM
            "options": {
                "num_ctx": config.NUM_CTX,
                "num_predict": num_predict,
                "temperature": config.TEMPERATURE,
                "top_p": getattr(config, "TOP_P", 0.9),
                "min_p": getattr(config, "MIN_P", 0.06),
                "repeat_penalty": config.REPEAT_PENALTY,
            },
        })
        resp.raise_for_status()
        return resp

    def check(self) -> None:
        """Проверяет, что Ollama жива и модель скачана; прогревает системный промпт и few-shot в VRAM."""
        import os
        tags = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=5).json()
        models = [m["name"] for m in tags.get("models", [])]
        # Если в Ollama уже собрана дообученная модель tolyan:8b, подхватываем её автоматически
        if "TOLYAN_MODEL" not in os.environ:
            for custom in getattr(config, "CUSTOM_MODEL_NAMES", ()):
                if custom in models:
                    config.OLLAMA_MODEL = custom
                    break
        if config.OLLAMA_MODEL not in models:
            raise RuntimeError(f"модель {config.OLLAMA_MODEL} не найдена в Ollama "
                               f"(есть: {', '.join(models) or 'ничего'})")
        # Прогреваем KV-кэш системного промпта и примеров диалога, чтобы первый ответ был быстрым.
        self._request(
            [*self._base_messages(), {"role": "user", "content": "Здарова, Толян"}],
            stream=False,
            num_predict=1,
        )

    def _trim_history(self) -> None:
        if len(self.history) < config.HISTORY_MAX:
            return
        keep = config.HISTORY_KEEP
        del self.history[:-keep]
        # Гарантируем, что история начинается с реплики user, а не разорванной пары.
        while self.history and self.history[0].get("role") != "user":
            self.history.pop(0)

    def reply_stream(self, text: str):
        """Генератор: отдаёт ответ кусками по предложению, пока модель ещё пишет дальше.

        Короткие вводные фразы (< MIN_CHUNK_CHARS) склеивает со следующим предложением,
        чтобы Толян не обрывал мысль на междометиях.
        """
        self._trim_history()
        if self.overheard:
            bg = " / ".join(self.overheard)
            self.overheard.clear()
            clean_msg = f"[Фон в канале: {bg}]\n{text}"
        else:
            clean_msg = text

        lore_hint = retrieve_lore(clean_msg)
        if lore_hint:
            prompt_msg = f"[Вспомнил прикол конфы: {lore_hint}]\n{clean_msg}"
        else:
            prompt_msg = clean_msg

        self.history.append({"role": "user", "content": prompt_msg})
        said: list[str] = []
        buf, sentences, cut = "", 0, False
        try:
            resp = self._request(
                [*self._base_messages(), *self.history],
                stream=True,
                num_predict=config.NUM_PREDICT,
            )
            with resp:
                for line in resp.iter_lines():
                    if not line:
                        continue
                    part = json.loads(line)
                    buf += part.get("message", {}).get("content", "")
                    # Сразу вырезаем ролевые действия *в звёздочках*, чтобы не засорять историю и TTS
                    buf = re.sub(r"\*[^*]+\*", " ", buf)
                    cut = part.get("done_reason") == "length"
                    pos = 0
                    for match in SENTENCE_END.finditer(buf):
                        chunk = buf[pos:match.end()].strip()
                        # Короткую присказку не считаем отдельным предложением — клеим дальше
                        if len(chunk) < MIN_CHUNK_CHARS:
                            continue
                        sentences += 1
                        pos = match.end()
                        said.append(chunk)
                        yield chunk
                        if sentences >= config.MAX_SENTENCES:
                            return  # выходим из with — Ollama перестаёт генерировать
                    buf = buf[pos:]
            tail = re.sub(r"\*[^*]*\*?", " ", buf).strip()
            if cut:  # упёрлись в лимит токенов посреди фразы
                if said and len(tail) < MIN_CHUNK_CHARS:
                    tail = ""
                else:  # рубим незавершённый хвост по последней запятой
                    head = tail.rsplit(",", 1)[0] if "," in tail else tail
                    tail = head.rstrip(" ,;:-—") + "."
            if tail and tail != ".":
                said.append(tail)
                yield tail
        finally:
            if said:
                # В долгую историю оставляем компактную реплику без длинной подсказки лора
                self.history[-1]["content"] = clean_msg
                self.history.append({"role": "assistant", "content": " ".join(said)})
            else:
                self.history.pop()

    def reply(self, text: str) -> str:
        return " ".join(self.reply_stream(text))


# ───────────────────────────── Рот ─────────────────────────────

_TTS_RULES = [
    (re.compile(pat, re.IGNORECASE), repl)
    for pat, repl in getattr(config, "TTS_PRONUNCIATION", ())
]


def clean_for_tts(text: str) -> str:
    """Убирает разметку и переводит английские слова/аббревиатуры в русскую фонетику для TTS."""
    text = re.sub(r"\*[^*]*\*", " ", text)
    text = re.sub(r"\[[^\]]*\]", " ", text)
    for regex, repl in _TTS_RULES:
        text = regex.sub(repl, text)
    text = re.sub(r"[^\w\s.,!?;:()\-—«»\"'%+]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def play_tts(text: str, sink_name: str, voice: str) -> int:
    """Синтезирует речь и сразу льёт её в указанный sink. Возвращает размер mp3 в байтах.

    Если сервис не отдал звук, бросает исключение edge-tts, ничего не проиграв.
    """
    procs = []

    async def stream() -> int:
        total = 0
        async for chunk in edge_tts.Communicate(text, voice, rate=config.TTS_RATE).stream():
            if chunk["type"] != "audio":
                continue
            if not procs:  # плеер поднимаем только когда звук реально пошёл
                decoder = subprocess.Popen(
                    ["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-f", "s16le",
                     "-ar", "48000", "-ac", "2", "pipe:1"],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE)
                player = subprocess.Popen(
                    ["paplay", "--raw", "--format=s16le", "--rate=48000", "--channels=2",
                     f"--latency-msec={config.PLAY_LATENCY_MS}",
                     f"--device={sink_name}", "--client-name=Tolyan"], stdin=decoder.stdout)
                decoder.stdout.close()
                procs.extend((decoder, player))
            procs[0].stdin.write(chunk["data"])
            total += len(chunk["data"])
        return total

    try:
        return asyncio.run(stream())
    finally:
        if procs:
            procs[0].stdin.close()
            for proc in procs:
                proc.wait()


class PiperTTS:
    """Локальный синтез: модель один раз грузится в память и отдаёт звук за доли секунды."""

    def __init__(self):
        from piper import PiperVoice, SynthesisConfig
        import os
        path = os.path.join(config.PIPER_DIR, f"{config.PIPER_VOICE}.onnx")
        if not os.path.exists(path):
            raise RuntimeError(f"нет файла голоса {path} (скачать: .venv/bin/python -m "
                               f"piper.download_voices {config.PIPER_VOICE} --data-dir voices)")
        self.name = f"piper {config.PIPER_VOICE}"
        self.voice = PiperVoice.load(path)
        self.syn = SynthesisConfig(length_scale=1.0 / float(config.PIPER_SPEED))
        for _ in self.voice.synthesize("Раз.", self.syn):  # прогрев, чтобы первая фраза не ждала
            pass

    def play(self, text: str, sink_name: str) -> int:
        player, total = None, 0
        try:
            for chunk in self.voice.synthesize(text, self.syn):
                if player is None:
                    player = subprocess.Popen(
                        ["paplay", "--raw", "--format=s16le", f"--rate={chunk.sample_rate}",
                         f"--channels={chunk.sample_channels}",
                         f"--latency-msec={config.PLAY_LATENCY_MS}",
                         f"--device={sink_name}", "--client-name=Tolyan"], stdin=subprocess.PIPE)
                player.stdin.write(chunk.audio_int16_bytes)
                total += len(chunk.audio_int16_bytes)
        finally:
            if player:
                player.stdin.close()
                player.wait()
        return total


piper_tts = None  # поднимается в init_tts()


def init_tts() -> str:
    """Готовит синтезатор и возвращает его название для вывода."""
    global piper_tts
    if config.TTS_ENGINE == "piper" and piper_tts is None:
        try:
            piper_tts = PiperTTS()
        except Exception as exc:
            console.print(f"[yellow]⚠ Piper не поднялся ({exc}) — говорю через edge-tts[/]")
    return piper_tts.name if piper_tts else f"edge-tts {config.TTS_VOICE}"


def switch_voice(target: str | None = None) -> str:
    """Переключает голос Толяна на лету (между dmitri, ruslan, denis и edge-tts)."""
    global piper_tts
    cycle = list(getattr(config, "PIPER_VOICES_CYCLE", ("ru_RU-dmitri-medium", "ru_RU-ruslan-medium", "ru_RU-denis-medium")))
    if target:
        t = target.strip().lower()
        if t == "edge":
            config.TTS_ENGINE = "edge"
            piper_tts = None
            return f"edge-tts {config.TTS_VOICE}"
        for v in cycle:
            if t in v.lower():
                config.TTS_ENGINE = "piper"
                config.PIPER_VOICE = v
                piper_tts = None
                return init_tts()
    # Циклическое переключение
    if config.TTS_ENGINE == "piper" and config.PIPER_VOICE in cycle:
        idx = (cycle.index(config.PIPER_VOICE) + 1) % len(cycle)
        config.PIPER_VOICE = cycle[idx]
    else:
        config.TTS_ENGINE = "piper"
        config.PIPER_VOICE = cycle[0]
    piper_tts = None
    return init_tts()


VOICES = [config.TTS_VOICE, *(v for v in config.TTS_FALLBACK_VOICES if v != config.TTS_VOICE)]
_voice_down: dict[str, float] = {}  # голос → до какого момента его не трогаем


def speak(text: str, sink_name: str) -> tuple[int, str]:
    """Говорит основным голосом. edge-tts иногда случайно не отдаёт звук, поэтому каждая
    попытка повторяется, и только потом берётся запасной голос. Голос, который не ответил,
    на TTS_VOICE_COOLDOWN_SEC выводится из игры — иначе каждая реплика ждала бы его заново."""
    if piper_tts:
        try:
            return piper_tts.play(text, sink_name), piper_tts.name
        except Exception as exc:
            console.print(f"[yellow]⚠ Piper споткнулся ({exc}) — эта фраза через edge-tts[/]")
    now = time.time()
    alive = [v for v in VOICES if _voice_down.get(v, 0) <= now] or VOICES
    error = None
    for voice in alive:
        for _ in range(config.TTS_ATTEMPTS):
            try:
                size = play_tts(text, sink_name, voice)
            except Exception as exc:
                error = exc
                continue
            return size, voice
        _voice_down[voice] = time.time() + config.TTS_VOICE_COOLDOWN_SEC
        console.print(f"[yellow]⚠ голос {voice} не отвечает — "
                      f"{config.TTS_VOICE_COOLDOWN_SEC // 60} мин говорю запасным[/]")
    raise RuntimeError(f"edge-tts не отдал звук ни одним голосом: {error}")


class Mouth:
    def __init__(self, muted: threading.Event):
        self.muted = muted
        self._aborted = threading.Event()
        self._player = None

    def interrupt(self) -> None:
        """Мгновенно прерывает текущую речь Толяна."""
        self._aborted.set()
        if self._player:
            try:
                self._player.terminate()
            except Exception:
                pass

    def say(self, chunks) -> None:
        """Озвучивает куски по мере поступления; генерация следующих предложений идёт параллельно озвучке."""
        self._aborted.clear()
        q: queue.Queue[str | None] = queue.Queue()
        gen_err: list[Exception] = []

        def producer():
            try:
                for chunk in chunks:
                    if self._aborted.is_set():
                        break
                    text = clean_for_tts(chunk)
                    if text:
                        q.put(text)
            except Exception as exc:
                gen_err.append(exc)
            finally:
                q.put(None)

        threading.Thread(target=producer, daemon=True).start()
        player = None
        try:
            while not self._aborted.is_set():
                text = q.get()
                if text is None or self._aborted.is_set():
                    break
                self.muted.set()
                if piper_tts:
                    try:
                        for audio_chunk in piper_tts.voice.synthesize(text, piper_tts.syn):
                            if self._aborted.is_set():
                                break
                            if player is None:
                                player = subprocess.Popen(
                                    ["paplay", "--raw", "--format=s16le",
                                     f"--rate={audio_chunk.sample_rate}",
                                     f"--channels={audio_chunk.sample_channels}",
                                     f"--latency-msec={config.PLAY_LATENCY_MS}",
                                     f"--device={config.MOUTH_SINK}", "--client-name=Tolyan"],
                                    stdin=subprocess.PIPE,
                                )
                                self._player = player
                            player.stdin.write(audio_chunk.audio_int16_bytes)
                        continue
                    except Exception as exc:
                        if self._aborted.is_set():
                            break
                        console.print(f"[yellow]⚠ Piper споткнулся ({exc}) — эта фраза через edge-tts[/]")
                speak(text, config.MOUTH_SINK)
            if gen_err and not self._aborted.is_set():
                raise gen_err[0]
        finally:
            self._player = None
            if player:
                try:
                    player.stdin.close()
                except Exception:
                    pass
                player.wait()
            if self.muted.is_set() and not self._aborted.is_set():
                time.sleep(config.POST_SPEAK_PAUSE)
            self.muted.clear()


# ───────────────────────────── Команды и отклик ─────────────────────────────

def _words(raw: str) -> set[str]:
    return {w.strip().lower().replace("ё", "е") for w in raw.split(",") if w.strip()}


def _has_phrase(norm_text: str, words_set: set[str], tokens: set[str]) -> bool:
    for item in words_set:
        if " " in item:
            if item in norm_text:
                return True
        elif item in tokens:
            return True
    return False


def mentioned_name(text: str) -> bool:
    """Проверяет, позвали ли Толяна по имени (с учётом падежей: Толяну, Толяна...)."""
    norm = text.lower().replace("ё", "е")
    said = re.findall(r"\w+", norm)
    wake = _words(config.WAKE_WORDS)
    names = wake if wake else _words(config.NAME_WORDS)
    return any(word.startswith(name) for word in said for name in names)


def voice_command(text: str, in_dialog: bool = False) -> str | None:
    """Распознаёт голосовые команды управления Толяном."""
    norm = text.lower().replace("ё", "е")
    tokens = set(re.findall(r"\w+", norm))
    named = mentioned_name(text)

    if named:
        if _has_phrase(norm, _words(config.PAUSE_WORDS), tokens):
            return "pause"
        if _has_phrase(norm, _words(config.RESUME_WORDS), tokens):
            return "resume"
        if _has_phrase(norm, _words(getattr(config, "FORGET_WORDS", "")), tokens):
            return "forget"
        if _has_phrase(norm, _words(getattr(config, "REMEMBER_WORDS", "")), tokens):
            return "remember"
        if _has_phrase(norm, _words(getattr(config, "VOICE_CHANGE_WORDS", "")), tokens):
            return "voice_change"
        if _has_phrase(norm, _words(getattr(config, "WINDOW_END_WORDS", "")), tokens):
            return "end_window"
    elif in_dialog:
        if _has_phrase(norm, _words(getattr(config, "REMEMBER_WORDS", "")), tokens):
            return "remember"
        if _has_phrase(norm, _words(getattr(config, "VOICE_CHANGE_WORDS", "")), tokens):
            return "voice_change"
        if _has_phrase(norm, _words(getattr(config, "WINDOW_END_WORDS", "")), tokens):
            return "end_window"
    return None


def show_cables(report) -> None:
    table = Table(title="🔌 Виртуальные кабели", title_justify="left")
    table.add_column("Устройство"); table.add_column("Описание"); table.add_column("Статус")
    for name, desc, status in report:
        color = "green" if status in ("создан", "уже был") else "yellow"
        table.add_row(name, desc, f"[{color}]{status}[/]")
    console.print(table)


def router_loop(stop: threading.Event) -> None:
    """Фоном следит за потоками Discord и пересаживает их на кабели Толяна."""
    connected = False
    while not stop.is_set():
        try:
            found, moved = audio.route_discord()
            if moved:
                console.print(f"[cyan]🔀 Discord пересажен на кабели Толяна (потоков: {moved})[/]")
            if found and not connected:
                console.print("[green]✅ Discord подключён: слышу канал и говорю в него[/]")
            elif connected and not found:
                console.print("[yellow]⚠ Потоки Discord пропали — зайди в голосовой канал[/]")
            connected = bool(found)
        except Exception as exc:
            console.print(f"[red]Маршрутизация: {exc}[/]")
        stop.wait(config.ROUTE_INTERVAL_SEC)


def collect_utterance_chunks(pending: queue.Queue, ear: Ear) -> list[bytes]:
    """Забирает фразу из очереди; если человек просто вдохнул и продолжает говорить — ждёт конец."""
    chunks = [pending.get(timeout=0.5)]
    while not pending.empty():
        chunks.append(pending.get_nowait())

    # Даём небольшую паузу: вдруг человек сделал вдох и прямо сейчас договаривает вторую часть мысли
    wait_sec = getattr(config, "FOLLOWUP_WAIT_SEC", 0.45)
    if wait_sec > 0 and not ear.is_speaking and pending.empty():
        time.sleep(wait_sec)
    if ear.is_speaking:
        deadline = time.time() + 3.5
        while ear.is_speaking and time.time() < deadline:
            time.sleep(0.05)
    while not pending.empty():
        chunks.append(pending.get_nowait())
    return chunks


def run(args) -> None:
    console.print(Panel.fit("[bold]ТОЛЯН[/] выходит на связь", border_style="magenta"))
    show_cables(audio.ensure_devices(monitor=args.monitor))

    brain = Brain()
    with console.status(f"🧠 Гружу {config.OLLAMA_MODEL} в память..."):
        brain.check()
    console.print(f"[green]🧠 Мозги на месте:[/] {config.OLLAMA_MODEL} (контекст {config.NUM_CTX})")
    with console.status("👂 Готовлю распознавание речи..."):
        stt = make_stt(args.stt)
    vad_name = "webrtcvad + порог громкости" if webrtcvad else "порог громкости"
    console.print(f"[green]👂 Уши:[/] {stt.name}, VAD: {vad_name}")
    with console.status("🗣 Готовлю голос..."):
        tts_name = init_tts()
    console.print(f"[green]🗣 Голос:[/] {tts_name}")

    win_sec = getattr(config, "ACTIVE_WINDOW_SEC", 25.0)
    if win_sec > 0:
        console.print(
            f"[green]💬 Режим диалога:[/] откликаюсь по имени + держу диалог [bold]{win_sec:.0f} с[/] "
            "после ответа (в фоне грею уши и иногда вставляю слово)"
        )
    elif win_sec == 0:
        console.print("[green]💬 Режим диалога:[/] отвечаю на все реплики подряд")
    else:
        console.print("[green]💬 Режим диалога:[/] отвечаю строго по имени в каждой фразе")

    console.print(
        "[green]🤫 Голосовые команды:[/] «Толян, помолчи» | «Толян, говори» | "
        "«Толян, забудь» | «Толян, запомни <факт>» | «Толян, смени голос» | «не тебе»"
    )
    console.print(
        "[green]⌨ Пульт в консоли:[/] [bold]!<текст>[/] — сказать дословно от лица Толяна | "
        "[bold]<текст>[/] — спросить Толяна | [bold]/voice[/] — сменить голос | "
        "[bold]/chime[/] — встрять в разговор | [bold].[/] или [bold]/stop[/] — прервать речь"
    )
    console.print(f"[dim]В Discord: устройство ввода — «{config.VIRT_MIC_DESC}», вывода — "
                  f"«{config.EAR_DESC}» (переключу сам, как только зайдёшь в канал). "
                  "Ctrl+C — выход.[/]\n")

    muted, stop = threading.Event(), threading.Event()
    status = console.status("[bold green]👂 Грею уши (зови «Толян»)...[/]")
    ear = Ear(muted, on_voice=lambda: status.update("[bold yellow]🎙 Слышу голос...[/]"))
    mouth = Mouth(muted)
    pending: queue.Queue = queue.Queue()
    console_q: queue.Queue[str] = queue.Queue()
    paused = False
    dialog_until = 0.0
    last_spoke_at = time.time()

    def idle_label() -> str:
        if paused:
            return "[bold dim]🤫 Молчу, жду «Толян, говори»...[/]"
        rem = dialog_until - time.time()
        if win_sec == 0:
            return "[bold green]👂 Слушаю...[/]"
        if rem > 0:
            return f"[bold green]💬 В диалоге — слушаю без имени (ещё {rem:.0f} с)...[/]"
        return "[bold green]👂 Грею уши (зови «Толян»)...[/]"

    def listen():
        for utterance in ear.utterances():
            while pending.qsize() >= 4:  # не копим старьё, если не успеваем
                pending.get_nowait()
            pending.put(utterance)
        stop.set()

    def console_input_loop():
        while not stop.is_set():
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                cmd = line.strip()
                if not cmd:
                    continue
                if cmd in (".", "/stop"):
                    mouth.interrupt()
                    console.print("[yellow]⏹ Речь прервана[/]")
                    continue
                console_q.put(cmd)
            except Exception:
                break

    threading.Thread(target=router_loop, args=(stop,), daemon=True).start()
    threading.Thread(target=listen, daemon=True).start()
    threading.Thread(target=console_input_loop, daemon=True).start()
    status.start()
    try:
        while not stop.is_set():
            forced_text: str | None = None
            if not console_q.empty():
                raw_cmd = console_q.get_nowait()
                if raw_cmd.startswith("!") or raw_cmd.lower().startswith("/say "):
                    say_txt = raw_cmd[1:].strip() if raw_cmd.startswith("!") else raw_cmd[5:].strip()
                    if say_txt:
                        console.print(f"[bold magenta]🎭 Толян (суфлёр):[/] {say_txt}")
                        status.update("[bold red]🗣 Толян говорит...[/]")
                        mouth.say([say_txt])
                        last_spoke_at = time.time()
                        if win_sec > 0:
                            dialog_until = time.time() + win_sec
                    status.update(idle_label())
                    continue
                if raw_cmd.lower().startswith("/voice"):
                    arg_v = raw_cmd[6:].strip() or None
                    new_v = switch_voice(arg_v)
                    console.print(f"[green]🗣 Голос переключён на:[/] {new_v}")
                    mouth.say([getattr(config, "VOICE_CHANGE_REPLY", "Голос переключён.")])
                    status.update(idle_label())
                    continue
                if raw_cmd.lower() == "/reset":
                    brain.reset()
                    console.print("[cyan]🧹 Память диалога сброшена[/]")
                    status.update(idle_label())
                    continue
                if raw_cmd.lower() == "/chime":
                    forced_text = "Толян, прокомментируй то, о чём сейчас говорили пацаны в канале."
                else:
                    forced_text = raw_cmd

            if forced_text is None:
                try:
                    chunks = collect_utterance_chunks(pending, ear)
                except queue.Empty:
                    status.update(idle_label())
                    continue

            try:
                if forced_text is not None:
                    text = forced_text
                else:
                    status.update("[bold blue]📝 Распознаю...[/]")
                    text = " ".join(filter(None, (stt.transcribe(c) for c in chunks)))
                    if not text:
                        continue

                in_dialog = (win_sec == 0) or (time.time() < dialog_until) or (forced_text is not None)
                command = voice_command(text, in_dialog=in_dialog) if forced_text is None else None

                if command == "pause" and not paused:
                    paused = True
                    dialog_until = 0.0
                    console.print(f"[bold cyan]🎧 Пацаны:[/] {text}")
                    console.print("[yellow]🤫 Толян замолчал[/]")
                    mouth.say([config.PAUSE_REPLY])
                    continue

                if command == "resume" and paused:
                    paused = False
                    if win_sec > 0:
                        dialog_until = time.time() + win_sec
                    console.print(f"[bold cyan]🎧 Пацаны:[/] {text}")
                    console.print("[green]🗣 Толян снова в разговоре[/]")
                    mouth.say([config.RESUME_REPLY])
                    if win_sec > 0:
                        dialog_until = time.time() + win_sec
                    continue

                if paused and forced_text is None:
                    continue

                if command == "forget":
                    brain.reset()
                    if win_sec > 0:
                        dialog_until = time.time() + win_sec
                    console.print(f"[bold cyan]🎧 Пацаны:[/] {text}")
                    console.print("[cyan]🧹 Толян сбросил память диалога[/]")
                    mouth.say([config.FORGET_REPLY])
                    if win_sec > 0:
                        dialog_until = time.time() + win_sec
                    continue

                if command == "remember":
                    saved = brain.remember(text)
                    if win_sec > 0:
                        dialog_until = time.time() + win_sec
                    console.print(f"[bold cyan]🎧 Пацаны:[/] {text}")
                    console.print(f"[green]📌 Записано в memory.json:[/] {saved}")
                    mouth.say([getattr(config, "REMEMBER_REPLY", "Записал на корочку!")])
                    if win_sec > 0:
                        dialog_until = time.time() + win_sec
                    continue

                if command == "voice_change":
                    new_v = switch_voice()
                    if win_sec > 0:
                        dialog_until = time.time() + win_sec
                    console.print(f"[bold cyan]🎧 Пацаны:[/] {text}")
                    console.print(f"[green]🗣 Голос переключён на:[/] {new_v}")
                    mouth.say([getattr(config, "VOICE_CHANGE_REPLY", "Прочистил горло!")])
                    if win_sec > 0:
                        dialog_until = time.time() + win_sec
                    continue

                if command == "end_window":
                    dialog_until = 0.0
                    brain.overhear(text)
                    console.print(f"[dim]👂 Пацаны (между собой): {text}[/]")
                    continue

                named = mentioned_name(text) or (forced_text is not None)
                should_reply = named or in_dialog
                chime = False

                if not should_reply and win_sec > 0:
                    # Иногда Толян сам вставляет свои 5 копеек в разговор пацанов
                    words_count = len(text.split())
                    cooldown = getattr(config, "RANDOM_CHIME_COOLDOWN_SEC", 90.0)
                    chance = getattr(config, "RANDOM_CHIME_CHANCE", 0.08)
                    if (words_count >= 5
                            and time.time() - last_spoke_at >= cooldown
                            and random.random() < chance):
                        should_reply = True
                        chime = True

                if not should_reply:
                    brain.overhear(text)
                    console.print(f"[dim]👂 (в фоне) Пацаны: {text}[/]")
                    continue

                tag = " [dim](встрял сам)[/]" if chime else ""
                console.print(f"[bold cyan]🎧 Пацаны:[/] {text}{tag}")
                status.update("[bold magenta]🧠 Толян думает...[/]")
                started = time.time()

                def spoken():
                    for chunk in brain.reply_stream(text):
                        console.print(f"[bold magenta]🗣 Толян:[/] {chunk} "
                                      f"[dim]({time.time() - started:.1f} с)[/]")
                        status.update("[bold red]🗣 Толян говорит...[/]")
                        yield chunk

                mouth.say(spoken())
                while not pending.empty():  # сбрасываем возможное эхо из канала во время речи Толяна
                    pending.get_nowait()
                last_spoke_at = time.time()
                if win_sec > 0 and (named or in_dialog):
                    dialog_until = time.time() + win_sec
            except Exception as exc:
                console.print(f"[red]Ошибка: {exc}[/]")
            finally:
                status.update(idle_label())
        console.print("[red]Запись с Bot_Ear оборвалась (parec завершился).[/]")
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        ear.stop()
        status.stop()
        console.print("\n[dim]Толян ушёл. Кабели оставлены (убрать: ./run.sh --cleanup).[/]")


# ───────────────────────────── Самопроверка ─────────────────────────────

def selftest(args) -> int:
    failed = 0

    def step(title, func):
        nonlocal failed
        try:
            detail = func()
            console.print(f"[green]✅ {title}[/] — {detail}")
        except Exception as exc:
            failed += 1
            console.print(f"[red]❌ {title}[/] — {exc}")

    def cables():
        show_cables(audio.ensure_devices(monitor=args.monitor))
        found, moved = audio.route_discord()
        return f"кабели на месте; потоков Discord: {found}, переключено: {moved}"

    def ollama():
        brain = Brain()
        brain.check()
        started = time.time()
        answer = brain.reply("Здарова, Толян, как сам?")
        return f"ответ за {time.time() - started:.1f} с: «{answer}»"

    def mouth_to_mic():
        rec = subprocess.Popen(record_cmd(config.VIRT_MIC), stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL)
        time.sleep(0.3)
        init_tts()
        size, voice = speak("Раз, два, проверка связи.", config.MOUTH_SINK)
        time.sleep(0.3)
        rec.terminate()
        recorded = rec.stdout.read()
        level = rms(recorded[: len(recorded) // 2 * 2])
        if size == 0:
            raise RuntimeError("синтезатор не вернул звук")
        if level < 50:
            raise RuntimeError(f"в {config.VIRT_MIC} тишина (RMS {level:.0f})")
        return (f"голос {voice}, синтезатор дал {size // 1024} КБ, "
                f"в {config.VIRT_MIC} пришёл звук (RMS {level:.0f})")

    def ear_loop():
        phrase = "Толян, здарова, как дела?"
        stt = make_stt(args.stt)
        ear = Ear(threading.Event())
        result: queue.Queue = queue.Queue()
        threading.Thread(target=lambda: result.put(next(ear.utterances(), b"")),
                         daemon=True).start()
        time.sleep(0.5)
        play_tts(phrase, config.EAR_SINK, voice="ru-RU-SvetlanaNeural")
        try:
            pcm = result.get(timeout=10)
        except queue.Empty:
            raise RuntimeError("VAD не поймал фразу в Bot_Ear") from None
        finally:
            ear.stop()
        text = stt.transcribe(pcm)
        if not text:
            raise RuntimeError("фраза поймана, но не распознана")
        return f"в уши сказано «{phrase}», {stt.name} услышал «{text}»"

    step("Кабели pactl", cables)
    step("Ollama", ollama)
    step("TTS → Bot_Mouth → VirtMic", mouth_to_mic)
    step("Bot_Ear → VAD → STT", ear_loop)
    console.print("[bold green]Всё работает.[/]" if not failed
                  else f"[bold red]Не прошло проверок: {failed}[/]")
    return 1 if failed else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Толян — голосовой пацан для Discord")
    parser.add_argument("--stt", choices=("google", "whisper"), default=config.STT_ENGINE,
                        help="движок распознавания речи")
    parser.add_argument("--monitor", action="store_true",
                        help="дублировать уши и рот Толяна в твои колонки/наушники")
    parser.add_argument("--test", action="store_true", help="самопроверка всех узлов и выход")
    parser.add_argument("--cleanup", action="store_true",
                        help="удалить виртуальные кабели и выйти")
    args = parser.parse_args()
    if args.cleanup:
        console.print(f"Удалено модулей: {audio.remove_devices()}")
    elif args.test:
        sys.exit(selftest(args))
    else:
        run(args)


if __name__ == "__main__":
    main()
