"""Fase 1: pindai kata OOV / dialek lokal dari data mentah.

Menghasilkan data/slang_candidates.csv untuk ditinjau manusia. Skrip ini
tidak mengubah slang_dict.py secara otomatis.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from slang_dict import NOISE_WORDS, SLANG_DICT
from text_mining_pipeline import find_raw_file, load_raw, normalize_text


def count_tokens(texts: list[str]) -> Counter:
    counter: Counter = Counter()
    for text in texts:
        for token in normalize_text(text).split():
            if token in NOISE_WORDS or token in SLANG_DICT or len(token) <= 2 or token.startswith('emoji'):
                continue
            counter[token] += 1
    return counter


def out_of_vocabulary(counter: Counter, known: set[str],
                      min_count: int = 3) -> list[tuple[str, int]]:
    rows = [(w, c) for w, c in counter.items() if w not in known and c >= min_count]
    return sorted(rows, key=lambda kv: (-kv[1], kv[0]))


def write_candidates(rows: list[tuple[str, int]], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["kata", "jumlah"]).to_csv(
        path, index=False, encoding="utf-8-sig")
    return path


def main() -> None:
    raw = load_raw(find_raw_file(Path("data")))
    counter = count_tokens(raw["text"].tolist())
    known = set(SLANG_DICT)
    try:
        from Sastrawi.StopWordRemover.StopWordRemoverFactory import \
            StopWordRemoverFactory
        known |= set(StopWordRemoverFactory().get_stop_words())
    except Exception:                                      # noqa: BLE001
        pass
    rows = out_of_vocabulary(counter, known, min_count=3)
    path = write_candidates(rows, Path("data/slang_candidates.csv"))
    print(f"{len(rows)} kandidat ditulis ke {path}")
    print(f"{len(raw)} komentar dipindai (termasuk data 2025)")
    for word, count in rows[:50]:
        print(f"  {count:4d}  {word}")


if __name__ == "__main__":
    main()

