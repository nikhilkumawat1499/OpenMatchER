import re
import unicodedata
from dataclasses import dataclass, field


DEFAULT_SUFFIXES = {
    "ltd",
    "limited",
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "llc",
    "pvt",
    "pvt ltd",
}


@dataclass(frozen=True)
class NormalizationConfig:
    lowercase: bool = True
    strip_punctuation: bool = True
    normalize_whitespace: bool = True
    stopwords: set[str] = field(default_factory=set)
    company_suffixes: set[str] = field(default_factory=lambda: set(DEFAULT_SUFFIXES))


def normalize_text(value: str, config: NormalizationConfig | None = None) -> str:
    config = config or NormalizationConfig()
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    if config.lowercase:
        text = text.lower()
    if config.strip_punctuation:
        text = re.sub(r"[^\w\s]", " ", text)
    tokens = [tok for tok in text.split() if tok not in config.stopwords]
    while tokens and " ".join(tokens[-2:]) in config.company_suffixes:
        tokens = tokens[:-2]
    while tokens and tokens[-1] in config.company_suffixes:
        tokens = tokens[:-1]
    text = " ".join(tokens)
    if config.normalize_whitespace:
        text = re.sub(r"\s+", " ", text).strip()
    return text
