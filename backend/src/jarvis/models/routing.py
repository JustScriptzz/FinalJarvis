"""Smart-routing heuristics and capability filtering."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..core.config import LLMProviderConfig, LLMRoutingConfig
from .types import ChatRequest

_CODE_FENCE = re.compile(r"```")
_CODE_KEYWORDS = re.compile(
    r"\b(implement|refactor|debug|traceback|stack trace|algorithm|complexity|"
    r"sql|regex|typescript|python|rust|dockerfile|kubernetes|migration|"
    r"unit test|pytest|compiler|race condition|deadlock|optimize)\b",
    re.IGNORECASE,
)
_REASONING_KEYWORDS = re.compile(
    r"\b(prove|derive|architecture|design|trade-?off|analy[sz]e|compare|"
    r"step[- ]by[- ]step|multi[- ]step|plan|strategy|root cause|why does|"
    r"evaluate|synthesi[sz]e|critique)\b",
    re.IGNORECASE,
)
_NUMBERED_STEP = re.compile(r"^\s*(?:\d+[.)]|[-*])\s+\S+", re.MULTILINE)


@dataclass(frozen=True)
class ComplexityAssessment:
    score: float
    profile: str
    reasons: list[str] = field(default_factory=list)
    requires_tools: bool = False
    requires_vision: bool = False
    requires_json: bool = False
    estimated_tokens: int = 0


def assess_complexity(request: ChatRequest, routing: LLMRoutingConfig) -> ComplexityAssessment:
    reasons: list[str] = []
    text = request.text()
    estimated_tokens = max(1, len(text) // 4)

    explicit = request.metadata.get("complexity")
    if isinstance(explicit, (int, float)) and 0 <= float(explicit) <= 1:
        score = float(explicit)
        reasons.append(f"explicit metadata complexity={score:.2f}")
    else:
        score = 0.0
        if estimated_tokens > 8_000:
            score += 0.15
            reasons.append("long prompt (>8k tokens)")
        if estimated_tokens > 24_000:
            score += 0.15
            reasons.append("very long prompt (>24k tokens)")
        if _CODE_FENCE.search(text):
            score += 0.18
            reasons.append("contains code")
        code_hits = len(set(match.group(0).lower() for match in _CODE_KEYWORDS.finditer(text)))
        if code_hits:
            score += min(0.18, 0.04 * code_hits)
            reasons.append(f"coding intent ({code_hits} signals)")
        reasoning_hits = len(
            set(match.group(0).lower() for match in _REASONING_KEYWORDS.finditer(text))
        )
        if reasoning_hits:
            score += min(0.20, 0.05 * reasoning_hits)
            reasons.append(f"reasoning intent ({reasoning_hits} signals)")
        steps = len(_NUMBERED_STEP.findall(text))
        if steps >= 3 or text.count("?") >= 3:
            score += 0.10
            reasons.append("multi-step request")
        if len(request.tools) >= 8:
            score += 0.08
            reasons.append("large toolset")
        if len(request.messages) >= 8:
            score += 0.08
            reasons.append("long conversation")
        score = min(1.0, score)

    if score < routing.fast_threshold:
        profile = "fast"
    elif score < routing.escalate_threshold:
        profile = "balanced"
    else:
        profile = "deep"

    return ComplexityAssessment(
        score=round(score, 3),
        profile=profile,
        reasons=reasons or ["routine request"],
        requires_tools=bool(request.tools),
        requires_vision=request.requires_vision(),
        requires_json=request.response_format.type != "text",
        estimated_tokens=estimated_tokens,
    )


def required_capabilities(assessment: ComplexityAssessment) -> set[str]:
    capabilities: set[str] = set()
    if assessment.requires_tools:
        capabilities.add("tools")
    if assessment.requires_vision:
        capabilities.add("vision")
    if assessment.requires_json:
        capabilities.add("json")
    return capabilities


def provider_capabilities(config: LLMProviderConfig) -> set[str]:
    capabilities: set[str] = set()
    if config.supports_tools:
        capabilities.add("tools")
    if config.supports_vision:
        capabilities.add("vision")
    if config.supports_json:
        capabilities.add("json")
    if config.supports_streaming:
        capabilities.add("streaming")
    return capabilities


def supports(config: LLMProviderConfig, required: set[str]) -> bool:
    return required.issubset(provider_capabilities(config))


def profile_chain(routing: LLMRoutingConfig, profile: str) -> list[str]:
    """Ordered candidates for a profile, degrading towards cheaper models."""
    chain: list[str] = []
    if profile == "deep":
        chain.extend(routing.profiles.get("deep", []))
        chain.extend(routing.profiles.get("balanced", []))
        chain.extend(routing.profiles.get("fast", []))
    elif profile == "balanced":
        chain.extend(routing.profiles.get("balanced", []))
        chain.extend(routing.profiles.get("fast", []))
    elif profile == "fast":
        chain.extend(routing.profiles.get("fast", []))
    else:
        chain.extend(routing.profiles.get(profile, []))
    chain.extend(routing.fallback)
    return list(dict.fromkeys(chain))


__all__ = [
    "ComplexityAssessment",
    "assess_complexity",
    "profile_chain",
    "provider_capabilities",
    "required_capabilities",
    "supports",
]
