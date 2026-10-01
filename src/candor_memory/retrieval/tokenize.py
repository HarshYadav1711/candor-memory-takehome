"""Small deterministic tokenizer for lexical retrieval."""

from __future__ import annotations

import re
import unicodedata


_TOKEN = re.compile(r"[^\W_]+(?:'[^\W_]+)?", re.UNICODE)


def tokenize(text: str) -> tuple[str, ...]:
    """Case-fold and extract Unicode word/number tokens.

    NFKC normalization and punctuation-based splitting make forms such as
    ``Route-Planner`` and ``route planner`` equivalent without applying a stem
    that could damage names, dates, identifiers, or measurements.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    normalized = unicodedata.normalize("NFKC", text).replace("’", "'").casefold()
    return tuple(match.group(0) for match in _TOKEN.finditer(normalized))

