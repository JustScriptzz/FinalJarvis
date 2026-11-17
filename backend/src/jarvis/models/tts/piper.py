"""Local Piper TTS via the ``piper`` CLI (raw PCM on stdout)."""

from __future__ import annotations

import asyncio
import shutil
import sys
import time
from pathlib import Path

from ...core.errors import ProviderUnavailableError
from ..audio import (
    estimate_duration,
    media_type_for_format,
    pcm16_to_wav,
)
from ..base import TTSProvider, elapsed_ms
from ..types import HealthStatus, SpeechRequest, SpeechResult


class PiperTTS(TTSProvider):
    type_name = "piper"

    def _binary(self) -> str | None:
        """Locate the ``piper`` executable.

        ``pip install piper-tts`` puts it in the environment's ``bin`` (or
        ``Scripts`` on Windows) directory, which is only on ``PATH`` when the
        environment is activated. The server is often started through the
        interpreter directly — ``.venv/bin/python -m jarvis...`` — so PATH check
        alone would report the voice as missing on a machine where it is
        installed. Look beside the running interpreter first.

        The Windows executable is ``piper.exe``; ``shutil.which`` finds it via
        ``PATHEXT``, but the beside-the-interpreter check has to name it exactly.
        """
        names = [self.config.binary]
        if not self.config.binary.lower().endswith(".exe"):
            names.append(self.config.binary + ".exe")
        for name in names:
            adjacent = Path(sys.executable).parent / name
            if adjacent.is_file():
                return str(adjacent)
        return shutil.which(self.config.binary)

    def _args(self) -> list[str]:
        binary = self._binary()
        if not binary:
            raise ProviderUnavailableError(
                f"Piper binary {self.config.binary!r} not found on PATH or beside "
                f"{sys.executable}; install it with: pip install piper-tts",
                provider_id=self.id,
            )
        model_path = self.config.model_path
        if not model_path or not Path(model_path).is_file():
            raise ProviderUnavailableError(
                f"Piper voice model not found: {model_path}", provider_id=self.id
            )
        args = [binary, "--model", model_path, "--output-raw"]
        if self.config.config_path:
            args.extend(["--config", self.config.config_path])
        if self.config.speaker is not None:
            args.extend(["--speaker", str(self.config.speaker)])
        for option, flag in (
            ("length_scale", "--length_scale"),
            ("noise_scale", "--noise_scale"),
            ("noise_w", "--noise_w"),
            ("sentence_silence", "--sentence_silence"),
            # Some Piper voices render hot and clip at full scale; the JARVIS
            # model does. ``volume`` is a plain multiplier applied by the CLI.
            ("volume", "--volume"),
        ):
            if option in self.config.options:
                args.extend([flag, str(self.config.options[option])])
        return args

    async def synthesize(self, request: SpeechRequest) -> SpeechResult:
        sample_rate = request.sample_rate or self.config.sample_rate or 22050
        process = await asyncio.create_subprocess_exec(
            *self._args(),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(request.text.encode("utf-8")),
                timeout=request.timeout_s or self.config.timeout_s,
            )
        except TimeoutError as exc:
            process.kill()
            await process.wait()
            raise ProviderUnavailableError(f"{self.id} timed out", provider_id=self.id) from exc
        if process.returncode != 0:
            message = stderr.decode("utf-8", "replace")[-1000:]
            raise ProviderUnavailableError(
                f"{self.id} failed (exit {process.returncode}): {message}",
                provider_id=self.id,
            )
        if not stdout:
            raise ProviderUnavailableError(f"{self.id} produced no audio", provider_id=self.id)

        audio_format = (request.format or self.config.format or "pcm").lower()
        if audio_format == "wav":
            audio = pcm16_to_wav(stdout, sample_rate)
            media_type = "audio/wav"
        else:
            audio = stdout
            audio_format = "pcm"
            media_type = media_type_for_format("pcm")
        return SpeechResult(
            audio=audio,
            media_type=media_type,
            format=audio_format,
            sample_rate=sample_rate,
            duration_s=estimate_duration(stdout, sample_rate),
            voice=request.voice or self.config.voice,
            provider_id=self.id,
            model=self.config.model or Path(self.config.model_path or "").name,
        )

    async def health(self) -> HealthStatus:
        if not self.config.enabled:
            return await super().health()
        started = time.perf_counter()
        if not self._binary():
            return HealthStatus(
                status="error",
                detail=f"Piper binary {self.config.binary!r} not found on PATH "
                f"or beside {sys.executable}; install it with: pip install piper-tts",
            )
        if not self.config.model_path or not await asyncio.to_thread(
            Path(self.config.model_path).is_file
        ):
            return HealthStatus(
                status="error", detail=f"voice model not found: {self.config.model_path}"
            )
        return HealthStatus(
            status="ok",
            detail="piper binary and voice model present",
            latency_ms=elapsed_ms(started),
        )


__all__ = ["PiperTTS"]
