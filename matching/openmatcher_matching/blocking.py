from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations


def soundex(value: str) -> str:
    value = value.upper()
    if not value:
        return ""
    codes = {"BFPV": "1", "CGJKQSXZ": "2", "DT": "3", "L": "4", "MN": "5", "R": "6"}
    lookup = {letter: digit for letters, digit in codes.items() for letter in letters}
    first, tail = value[0], value[1:]
    digits = [lookup.get(ch, "") for ch in tail]
    compact = []
    previous = lookup.get(first, "")
    for digit in digits:
        if digit != previous and digit:
            compact.append(digit)
        previous = digit
    return (first + "".join(compact) + "000")[:4]


@dataclass(frozen=True)
class BlockingRule:
    kind: str = "prefix"
    size: int = 4


def block_records(records: list[dict], field: str, rules: list[BlockingRule]) -> list[tuple[int, int]]:
    buckets: dict[str, set[int]] = defaultdict(set)
    for idx, row in enumerate(records):
        value = str(row.get(field, ""))
        tokens = value.split()
        for rule in rules:
            if rule.kind == "prefix":
                buckets[f"p:{value[:rule.size]}"].add(idx)
            elif rule.kind == "phonetic":
                buckets[f"s:{soundex(value)}"].add(idx)
            elif rule.kind == "ngram":
                grams = {value[i : i + rule.size] for i in range(max(1, len(value) - rule.size + 1))}
                for gram in grams:
                    buckets[f"n:{gram}"].add(idx)
            elif rule.kind == "token":
                for token in tokens:
                    buckets[f"t:{token}"].add(idx)
            else:
                raise ValueError(f"Unsupported blocking rule: {rule.kind}")
    pairs: set[tuple[int, int]] = set()
    for ids in buckets.values():
        if len(ids) > 1:
            pairs.update(tuple(sorted(pair)) for pair in combinations(ids, 2))
    return sorted(pairs)

