"""Виртуальные кабели и маршрутизация Discord через pactl (PipeWire/PulseAudio)."""
import json
import os
import subprocess

import config

_ENV = {**os.environ, "LC_ALL": "C"}


def pactl(*args: str) -> str:
    res = subprocess.run(["pactl", *args], capture_output=True, text=True, env=_ENV)
    if res.returncode != 0:
        raise RuntimeError(f"pactl {' '.join(args)}: {res.stderr.strip()}")
    return res.stdout.strip()


def pactl_json(*args: str) -> list[dict]:
    return json.loads(pactl("--format=json", *args) or "[]")


def _modules() -> list[tuple[str, str, str]]:
    """[(индекс, имя, аргументы)] — в JSON-выводе pipewire-pulse аргументов нет, берём текст."""
    rows = []
    for line in pactl("list", "short", "modules").splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            rows.append((parts[0], parts[1], parts[2] if len(parts) > 2 else ""))
    return rows


def _names(kind: str) -> dict[str, int]:
    """{имя: индекс} для sinks / sources."""
    return {item["name"]: int(item["index"]) for item in pactl_json("list", kind)}


def ensure_devices(monitor: bool = False) -> list[tuple[str, str, str]]:
    """Создаёт недостающие кабели. Возвращает [(имя, описание, 'создан' | 'уже был')]."""
    report = []
    sinks = _names("sinks")
    for name, desc in ((config.EAR_SINK, config.EAR_DESC), (config.MOUTH_SINK, config.MOUTH_DESC)):
        if name in sinks:
            report.append((name, desc, "уже был"))
            continue
        pactl("load-module", "module-null-sink", f"sink_name={name}",
              f"sink_properties=device.description={desc}")
        report.append((name, desc, "создан"))

    if config.VIRT_MIC in _names("sources"):
        report.append((config.VIRT_MIC, config.VIRT_MIC_DESC, "уже был"))
    else:
        pactl("load-module", "module-remap-source", f"master={config.MOUTH_SINK}.monitor",
              f"source_name={config.VIRT_MIC}",
              f"source_properties=device.description={config.VIRT_MIC_DESC}")
        report.append((config.VIRT_MIC, config.VIRT_MIC_DESC, "создан"))

    # WirePlumber может создать кабель с чужой громкостью (например, 19% как у микрофона).
    for sink in (config.EAR_SINK, config.MOUTH_SINK):
        pactl("set-sink-volume", sink, "100%")
        pactl("set-sink-mute", sink, "0")
    pactl("set-source-volume", config.VIRT_MIC, "100%")
    pactl("set-source-mute", config.VIRT_MIC, "0")

    if monitor:
        report += _ensure_monitor_loopbacks()
    return report


def _ensure_monitor_loopbacks() -> list[tuple[str, str, str]]:
    """Дублирует уши и рот Толяна в твои колонки/наушники, чтобы ты тоже всё слышал."""
    default_sink = pactl("get-default-sink")
    if default_sink in (config.EAR_SINK, config.MOUTH_SINK):
        return [("прослушка", "в колонки", "пропущена: устройство по умолчанию — кабель Толяна")]
    loaded = " ".join(args for _, name, args in _modules() if name == "module-loopback")
    report = []
    for sink in (config.EAR_SINK, config.MOUTH_SINK):
        source = f"{sink}.monitor"
        if f"source={source}" in loaded:
            report.append((f"{sink} → колонки", "прослушка", "уже был"))
            continue
        pactl("load-module", "module-loopback", f"source={source}", f"sink={default_sink}",
              "latency_msec=60")
        report.append((f"{sink} → колонки", "прослушка", "создан"))
    return report


def _is_discord(props: dict) -> bool:
    text = " ".join(str(props.get(key, "")) for key in (
        "application.name", "application.process.binary", "media.name", "node.name")).lower()
    return any(word in text for word in config.DISCORD_KEYWORDS)


def route_discord() -> tuple[int, int]:
    """Вывод Discord → уши Толяна, ввод Discord ← микрофон Толяна.

    Возвращает (сколько потоков Discord найдено, сколько из них пришлось переключить).
    """
    ear = _names("sinks").get(config.EAR_SINK)
    mic = _names("sources").get(config.VIRT_MIC)
    found = moved = 0
    for kind, key, target_idx, target, cmd in (
        ("sink-inputs", "sink", ear, config.EAR_SINK, "move-sink-input"),
        ("source-outputs", "source", mic, config.VIRT_MIC, "move-source-output"),
    ):
        for stream in pactl_json("list", kind):
            if not _is_discord(stream.get("properties", {})):
                continue
            found += 1
            if stream.get(key) == target_idx:
                continue
            try:
                pactl(cmd, str(stream["index"]), target)
                moved += 1
            except RuntimeError:
                pass  # поток мог исчезнуть между list и move
    return found, moved


def remove_devices() -> int:
    """Выгружает все модули, созданные для Толяна. Возвращает их количество."""
    marks = (f"sink_name={config.EAR_SINK}", f"sink_name={config.MOUTH_SINK}",
             f"source_name={config.VIRT_MIC}", f"source={config.EAR_SINK}.monitor",
             f"source={config.MOUTH_SINK}.monitor")
    removed = 0
    # Сначала зависимые модули (loopback, remap), потом сами sinks.
    for index, _, args in reversed(_modules()):
        if any(mark in args for mark in marks):
            try:
                pactl("unload-module", index)
                removed += 1
            except RuntimeError:
                pass
    return removed
