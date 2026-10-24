"""Masks personal data and secrets in captured prompts and responses before they leave the process."""

from __future__ import annotations

import re
from typing import Callable, Optional, Sequence, Tuple, Union

_SECRET = re.compile(r"\b(?:sk|pk|rk|om_live|om_test)[-_][A-Za-z0-9_\-]{16,}\b|\bBearer\s+[A-Za-z0-9._\-]{16,}")
_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_CARD = re.compile(r"(?<![\d])(?:\d[ \-]?){12,18}\d(?![\d])")
_PHONE = re.compile(r"(?<![\w+])\+?\(?\d[\d\s().\-]{7,}\d(?![\w])")


def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def _card(m: "re.Match[str]") -> str:
    digits = re.sub(r"\D", "", m.group(0))
    return "[CARD]" if 13 <= len(digits) <= 19 and _luhn_ok(digits) else m.group(0)


def _phone(m: "re.Match[str]") -> str:
    digits = re.sub(r"\D", "", m.group(0))
    return "[PHONE]" if 9 <= len(digits) <= 15 else m.group(0)


Pattern = Tuple[Union[str, "re.Pattern[str]"], str]


class Redactor:
    def __init__(self, enabled: bool = True, extra: Sequence[Pattern] = (), custom: Optional[Callable[[str], str]] = None):
        self.enabled = enabled
        self.extra = [(re.compile(p) if isinstance(p, str) else p, r) for p, r in extra]
        self.custom = custom

    def __call__(self, text: str) -> str:
        if not text:
            return text
        if self.enabled:
            text = _SECRET.sub("[SECRET]", text)
            text = _EMAIL.sub("[EMAIL]", text)
            text = _CARD.sub(_card, text)
            text = _PHONE.sub(_phone, text)
        for pat, repl in self.extra:
            text = pat.sub(repl, text)
        if self.custom is not None:
            text = self.custom(text)
        return text
