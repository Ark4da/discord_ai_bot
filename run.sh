#!/usr/bin/env bash
# Запуск Толяна. Примеры:
#   ./run.sh                 обычный запуск
#   ./run.sh --monitor       ещё и слышать канал и Толяна в своих колонках
#   ./run.sh --stt whisper   распознавание офлайн (faster-whisper на CPU)
#   ./run.sh --test          самопроверка
#   ./run.sh --cleanup       убрать виртуальные кабели
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

for tool in pactl parec paplay ffmpeg; do
    command -v "$tool" >/dev/null || {
        echo "Не найден $tool. Поставь: sudo dnf install -y ffmpeg pulseaudio-utils" >&2
        exit 1
    }
done

if [ ! -x .venv/bin/python ]; then
    echo "Создаю виртуальное окружение..."
    python3 -m venv .venv
    .venv/bin/pip install -q --upgrade pip
    .venv/bin/pip install -q -r requirements.txt
fi

mkdir -p voices
if [ ! -f voices/ru_RU-dmitri-medium.onnx ]; then
    echo "Скачиваю локальный голос Piper (ru_RU-dmitri-medium)..."
    .venv/bin/python -m piper.download_voices ru_RU-dmitri-medium --data-dir voices || true
fi

if ! curl -s -m 3 http://127.0.0.1:11434/api/tags >/dev/null; then
    if command -v ollama >/dev/null; then
        echo "Запускаю Ollama с оптимизацией под 4 ГБ VRAM (Flash Attention + q8_0 KV-кэш)..."
        OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 nohup ollama serve >/tmp/ollama-tolyan.log 2>&1 &
        sleep 2
    fi
    if ! curl -s -m 3 http://127.0.0.1:11434/api/tags >/dev/null; then
        echo "Ollama не отвечает на 127.0.0.1:11434. Запусти: OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve" >&2
        exit 1
    fi
fi

exec .venv/bin/python main.py "$@"
