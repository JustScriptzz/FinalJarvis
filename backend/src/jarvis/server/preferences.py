"""User tuning preferences.

Choosing a core is only half of the decision: a local Ollama server serves many
models, a TTS server serves many voices, and the same person may want a brisker
delivery in the evening than at 3 a.m. None of that belongs in ``config.yaml``
(the editor's file) or in the setup record (which core is engaged), so it lives
here, in one small record persisted in the settings table.

Every field is optional. ``None`` means "use whatever the provider is configured
with", which keeps a fresh install working with zero tuning while refusing to
invent a choice the user never made.
"""

from __future__ import annotations

import copy
from typing import Any

PREFERENCES_KEY = "preferences"

# Generic sample sentence for the voice preview button. Short enough to be
# cheap on a paid provider and affordable on a CPU-only cloning core, where a
# preview costs minutes of computation per second of audio, while still long
# enough to judge delivery.
PREVIEW_TEXT = "Good evening, sir. All systems are nominal."


DEFAULTS: dict[str, Any] = {
    "llm": {
        # Model name inside the selected provider, e.g. "qwen2.5:14b".
        "model": None,
        # The core this choice was made on. A model name is core-specific, so
        # switching cores clears it rather than sending an unknown name.
        "provider": None,
        "temperature": 0.65,
        "max_tokens": 2048,
    },
    "stt": {
        "model": None,
        "provider": None,
        "language": None,
    },
    "tts": {
        # Server voice id, or a browser SpeechSynthesis voice URI. They are
        # separate fields on purpose: a Kokoro voice id is meaningless to the
        # browser engine, and switching cores must not carry a stale name over.
        "voice": None,
        "browser_voice": None,
        "provider": None,
        "rate": 1.0,
        "pitch": 1.0,
        "speed": 1.0,
    },
}

_RANGES: dict[tuple[str, str], tuple[float, float]] = {
    ("llm", "temperature"): (0.0, 2.0),
    ("llm", "max_tokens"): (1.0, 32768.0),
    ("tts", "rate"): (0.5, 2.0),
    ("tts", "pitch"): (0.5, 2.0),
    ("tts", "speed"): (0.25, 4.0),
}

_TEXT_LIMITS: dict[tuple[str, str], int] = {
    ("llm", "model"): 200,
    ("llm", "provider"): 60,
    ("stt", "model"): 200,
    ("stt", "provider"): 60,
    ("stt", "language"): 20,
    ("tts", "voice"): 200,
    ("tts", "browser_voice"): 400,
    ("tts", "provider"): 60,
}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _clean_text(value: Any, limit: int) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text[:limit]


def normalize(raw: Any) -> dict[str, Any]:
    """Return a complete, validated preference record.

    Unknown keys are dropped and out-of-range numbers are clamped rather than
    rejected: a stale value from an older build must not stop the assistant from
    booting. Invalid types fall back to the default for that field.
    """
    record = copy.deepcopy(DEFAULTS)
    if not isinstance(raw, dict):
        return record
    for section, fields in DEFAULTS.items():
        incoming = raw.get(section)
        if not isinstance(incoming, dict):
            continue
        for field, fallback in fields.items():
            if field not in incoming:
                continue
            value = incoming[field]
            if (section, field) in _TEXT_LIMITS:
                # An empty string is a meaningful "no override".
                record[section][field] = _clean_text(value, _TEXT_LIMITS[(section, field)])
                continue
            low, high = _RANGES[(section, field)]
            try:
                number = float(value)
            except (TypeError, ValueError):
                record[section][field] = fallback
                continue
            record[section][field] = round(_clamp(number, low, high), 4)
    return record


def merge(current: dict[str, Any], patch: Any) -> dict[str, Any]:
    """Overlay a partial update on the stored record and re-validate it."""
    if not isinstance(patch, dict):
        return normalize(current)
    combined = normalize(current)
    for section, fields in DEFAULTS.items():
        incoming = patch.get(section)
        if not isinstance(incoming, dict):
            continue
        for field in fields:
            if field not in incoming:
                continue
            # ``None`` explicitly clears an override; absent leaves it alone.
            combined[section][field] = incoming[field]
    return normalize(combined)


def reconcile(
    preferences: dict[str, Any], selection: dict[str, Any]
) -> tuple[dict[str, Any], bool]:
    """Drop model/voice choices that belong to a core which is no longer engaged.

    A model name like ``qwen2.5:14b`` is meaningful only on the endpoint it came
    from, and a Kokoro voice id means nothing to ElevenLabs. Each choice records
    the core it was made on; when that core changes, the stale name is cleared
    instead of being sent to a server that has never heard of it.

    Returns the record and whether anything changed, so the caller only writes
    to the database when it must.
    """
    record = normalize(preferences)
    changed = False
    for kind in ("llm", "tts", "stt"):
        section = record[kind]
        owner = section.get("provider")
        current = selection.get(kind)
        if not owner:
            continue
        fields = {
            "llm": ("model",),
            "tts": ("voice",),
            "stt": ("model", "language"),
        }[kind]
        if current is None:
            # No core of this kind is engaged (keyboard-only input, for
            # instance): its model and language are meaningless. Delivery
            # settings are not core-specific and survive.
            for field in fields:
                if section.get(field) is not None:
                    section[field] = None
                    changed = True
            section["provider"] = None
            changed = True
            continue
        if owner == current:
            continue
        for field in fields:
            if section.get(field) is not None:
                section[field] = None
                changed = True
        section["provider"] = current
        changed = True
    return record, changed


def llm_request_options(preferences: dict[str, Any]) -> dict[str, Any]:
    """Keyword arguments for ``ChatRequest`` derived from the preferences."""
    llm = preferences.get("llm") or {}
    return {
        "model": llm.get("model"),
        "temperature": llm.get("temperature"),
        "max_tokens": int(llm.get("max_tokens") or 2048),
    }


def tts_request_options(preferences: dict[str, Any]) -> dict[str, Any]:
    """Voice and speed for a server-side synthesis request."""
    tts = preferences.get("tts") or {}
    return {
        "voice": tts.get("voice"),
        "speed": tts.get("speed"),
    }


__all__ = [
    "DEFAULTS",
    "PREFERENCES_KEY",
    "PREVIEW_TEXT",
    "llm_request_options",
    "merge",
    "normalize",
    "reconcile",
    "tts_request_options",
]


def stt_request_options(preferences: dict[str, Any]) -> dict[str, Any]:
    """Model and language for a transcription request.

    Both are hints the core may ignore: a local Whisper build has its own model
    loaded, and a server that cannot detect language needs the hint to do its
    best work. Sending them is still the difference between the user's choice
    mattering and a form that writes to a file nobody reads.
    """
    stt = preferences.get("stt") or {}
    return {
        "model": stt.get("model"),
        "language": stt.get("language"),
    }
