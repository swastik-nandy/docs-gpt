from __future__ import annotations

import re


_MEMORY_TRIGGER_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bremember\b",
        r"\bearlier\b",
        r"\blast time\b",
        r"\bprevious(?:ly)?\b",
        r"\bagain\b",
        r"\bsame project\b",
        r"\bmy setup\b",
        r"\bmy preference\b",
        r"\bas i said\b",
        r"\bwe discussed\b",
        r"\byou know\b",
        r"\bfrom now on\b",
        r"\bgoing forward\b",
    )
)


def should_retrieve_long_term_memory(user_message: str) -> bool:
    message = user_message.strip()
    if not message:
        return False

    return any(pattern.search(message) for pattern in _MEMORY_TRIGGER_PATTERNS)


def should_consider_memory_write(user_message: str) -> bool:
    message = user_message.strip()
    if not message:
        return False

    durable_signals = (
        "remember",
        "from now on",
        "going forward",
        "my preference",
        "i prefer",
        "we use",
        "my project",
        "our project",
    )

    lowered = message.lower()
    return any(signal in lowered for signal in durable_signals)
