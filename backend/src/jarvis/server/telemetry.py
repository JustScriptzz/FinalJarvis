"""Host telemetry for the JARVIS HUD.

Everything here is read-only and best-effort: the assistant may run on a
laptop, a headless server or Termux, so unavailable sensors degrade to
``None`` instead of breaking the dashboard.
"""

from __future__ import annotations

import asyncio
import os
import platform
import socket
import sys
import time
from collections import deque
from datetime import UTC, datetime
from typing import Any

import psutil


def _safe(call: Any, default: Any = None) -> Any:
    try:
        return call()
    except (psutil.Error, OSError, NotImplementedError, AttributeError, ValueError):
        return default


class TelemetryService:
    """Samples system state and keeps a short rolling history."""

    def __init__(self, history_size: int = 120) -> None:
        self._history: deque[dict[str, Any]] = deque(maxlen=history_size)
        self._last_net: tuple[float, int, int] | None = None
        self._host = self._host_info()
        self._sampling = asyncio.Lock()
        # Prime the non-blocking CPU counters.
        psutil.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None, percpu=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def sample(self) -> dict[str, Any]:
        async with self._sampling:
            snapshot = await asyncio.to_thread(self._sample_sync)
        compact = {
            "timestamp": snapshot["timestamp"],
            "cpu": snapshot["cpu"]["percent"],
            "memory": snapshot["memory"]["percent"],
            "disk": snapshot["disk"]["percent"],
            "net_rx": snapshot["network"]["recv_rate"],
            "net_tx": snapshot["network"]["send_rate"],
        }
        self._history.append(compact)
        return snapshot

    def history(self) -> list[dict[str, Any]]:
        return list(self._history)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def _sample_sync(self) -> dict[str, Any]:
        now = time.time()
        cpu_percent = float(_safe(psutil.cpu_percent, 0.0) or 0.0)
        per_core_raw = _safe(lambda: psutil.cpu_percent(interval=None, percpu=True), []) or []
        per_core = [float(value) for value in per_core_raw]
        memory = _safe(psutil.virtual_memory)
        swap = _safe(psutil.swap_memory)
        disk = _safe(lambda: psutil.disk_usage(self._host["root"]))
        net = _safe(psutil.net_io_counters)

        send_rate = recv_rate = 0.0
        if net is not None and self._last_net is not None:
            last_time, last_sent, last_recv = self._last_net
            elapsed = max(0.001, now - last_time)
            send_rate = max(0.0, (net.bytes_sent - last_sent) / elapsed)
            recv_rate = max(0.0, (net.bytes_recv - last_recv) / elapsed)
        if net is not None:
            self._last_net = (now, net.bytes_sent, net.bytes_recv)

        data: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            "epoch": now,
            "cpu": {
                "percent": round(cpu_percent, 1),
                "per_core": [round(value, 1) for value in per_core],
                "count": psutil.cpu_count() or 1,
                "load": [round(value, 2) for value in _safe(os.getloadavg, (0.0, 0.0, 0.0))],
            },
            "memory": {
                "total": getattr(memory, "total", 0),
                "used": getattr(memory, "used", 0),
                "available": getattr(memory, "available", 0),
                "percent": round(float(getattr(memory, "percent", 0.0)), 1),
                "swap_total": getattr(swap, "total", 0),
                "swap_used": getattr(swap, "used", 0),
                "swap_percent": round(float(getattr(swap, "percent", 0.0)), 1),
            },
            "disk": {
                "total": getattr(disk, "total", 0),
                "used": getattr(disk, "used", 0),
                "free": getattr(disk, "free", 0),
                "percent": round(float(getattr(disk, "percent", 0.0)), 1),
                "mount": self._host["root"],
            },
            "network": {
                "sent": getattr(net, "bytes_sent", 0),
                "recv": getattr(net, "bytes_recv", 0),
                "send_rate": round(send_rate, 1),
                "recv_rate": round(recv_rate, 1),
            },
            "processes": self._processes(),
            "host": dict(self._host),
            "uptime_s": round(now - (_safe(psutil.boot_time, now) or now), 1),
            "battery": self._battery(),
            "temperatures": self._temperatures(),
        }
        return data

    def _processes(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        try:
            for process in psutil.process_iter(
                ["pid", "name", "cpu_percent", "memory_percent", "username"]
            ):
                info = process.info
                items.append(
                    {
                        "pid": info.get("pid"),
                        "name": (info.get("name") or "?")[:28],
                        "cpu": round(float(info.get("cpu_percent") or 0.0), 1),
                        "memory": round(float(info.get("memory_percent") or 0.0), 2),
                        "user": info.get("username") or "",
                    }
                )
        except (psutil.Error, OSError):
            return []
        items.sort(key=lambda item: (item["cpu"], item["memory"]), reverse=True)
        return items[:6]

    def _battery(self) -> dict[str, Any] | None:
        battery = _safe(psutil.sensors_battery)
        if battery is None:
            return None
        return {
            "percent": round(float(battery.percent), 1),
            "plugged": bool(battery.power_plugged),
            "seconds_left": battery.secsleft
            if isinstance(battery.secsleft, int) and battery.secsleft >= 0
            else None,
        }

    def _temperatures(self) -> dict[str, float] | None:
        sensors = _safe(psutil.sensors_temperatures, {}) or {}
        result: dict[str, float] = {}
        for name, entries in sensors.items():
            for entry in entries:
                if entry.current is not None:
                    result[name] = round(float(entry.current), 1)
                    break
        return result or None

    @staticmethod
    def _host_info() -> dict[str, Any]:
        root = "/" if os.name != "nt" else "C:\\"
        try:
            hostname = socket.gethostname()
        except OSError:
            hostname = "jarvis-host"
        try:
            address = socket.gethostbyname(hostname)
        except OSError:
            address = "127.0.0.1"
        return {
            "hostname": hostname,
            "address": address,
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor() or platform.machine(),
            "python": sys.version.split()[0],
            "pid": os.getpid(),
            "root": root,
            "boot_time": datetime.fromtimestamp(psutil.boot_time(), UTC).isoformat(
                timespec="seconds"
            ),
        }
