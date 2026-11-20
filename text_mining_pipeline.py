"""Pipeline text mining komentar TikTok MBG Blitar 2026.

Alur: ingest -> filter -> preprocessing -> pelabelan hybrid -> split -> ekspor
-> pelatihan model -> evaluasi -> uji periode.

Semua fungsi di sini murni dan bisa diuji tanpa Streamlit.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from slang_dict import (EMOJI_MAP, NEGATION_WORDS, NOISE_WORDS, SLANG_DICT)

LOGGER = logging.getLogger(__name__)

DATA_DIR = Path("data")
INSET_BASE = "https://raw.githubusercontent.com/fajri91/InSet/master"
LABEL_ORDER = ["Negatif", "Netral", "Positif"]

COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "text": ("text", "comment", "komentar", "caption"),
    "date": ("createtimeiso", "createtime", "date", "tanggal", "timestamp"),
    "likes": ("diggcount", "likes", "like", "suka"),
    "user": ("uniqueid", "username", "user", "author", "pengguna"),
    "video_url": ("videoweburl", "url", "video_url", "link"),
}

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+|#\w+")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")
REPEAT_RE = re.compile(r"(.)\1{2,}")


# --------------------------------------------------------------------------
# Ingest
# --------------------------------------------------------------------------
def find_raw_file(data_dir: Path = DATA_DIR) -> Path:
    """Cari berkas mentah: JSON diprioritaskan, lalu CSV. Hasil turunan dilewati."""
    data_dir = Path(data_dir)
    candidates = []
    for path in data_dir.iterdir() if data_dir.exists() else []:
        name = path.name.lower()
        if path.suffix.lower() not in (".json", ".csv", ".jsonl"):
            continue
        if name.startswith(("dataset_", "validasi_", "slang_candidates")):
            continue
        candidates.append(path)
    if not candidates:
        raise FileNotFoundError(
            f"Tidak ada berkas .json/.csv di {data_dir}. Letakkan data mentah di sana."
        )
    rank = {".json": 0, ".jsonl": 1, ".csv": 2}
    candidates.sort(key=lambda p: (0 if p.name.lower().startswith("raw") else 1,
                                   rank.get(p.suffix.lower(), 9),
                                   p.name))
    return candidates[0]


def _read_json(path: Path):
    raw = path.read_text(encoding="utf-8-sig")
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            for key in ("items", "data", "results"):
                if isinstance(payload.get(key), list):
                    return payload[key]
            return []
        return payload
    except json.JSONDecodeError:
        return [json.loads(line) for line in raw.splitlines() if line.strip()]


def _is_number(value) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def load_raw(path: Path, text_col: str | None = None,
             date_col: str | None = None) -> pd.DataFrame:
    """Baca JSON/CSV Apify dan normalisasi nama kolom.

    `text_col` / `date_col` menimpa deteksi otomatis bila pengguna memilih
    kolom lain lewat sidebar.
    """
    path = Path(path)
    if path.suffix.lower() in (".json", ".jsonl"):
        records = _read_json(path)
        frame = pd.DataFrame(records)
    else:
        frame = pd.read_csv(path, encoding="utf-8-sig")
    frame.columns = [str(c).replace("\ufeff", "").strip() for c in frame.columns]

    lowered = {c.lower(): c for c in frame.columns}
    resolved: dict[str, str | None] = {}
    for target, aliases in COLUMN_ALIASES.items():
        resolved[target] = next(
            (lowered[a] for a in aliases if a in lowered), None
        )
    if text_col and text_col in frame.columns:
        resolved["text"] = text_col
    if date_col and date_col in frame.columns:
        resolved["date"] = date_col
        lowered["_override_date"] = date_col

    if resolved["text"] is None:
        raise ValueError(
            f"Kolom teks tidak ditemukan. Kolom tersedia: {list(frame.columns)}"
        )

    out = pd.DataFrame()
    out["text"] = frame[resolved["text"]].fillna("").astype(str).str.strip()

    # --- tanggal: createTimeISO lebih dulu, lalu createTime (epoch) ---
    date_series = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    date_candidates = (["_override_date"] if "_override_date" in lowered else []) + \
        ["createtimeiso", "createtime", "date", "tanggal", "timestamp"]
    for candidate in date_candidates:
        col = lowered.get(candidate)
        if col is None:
            continue
        values = frame[col]
        numeric = values.map(_is_number)
        parsed = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
        if numeric.all() and len(values):
            nums = pd.to_numeric(values, errors="coerce")
            if nums.dropna().gt(1e11).all():          # epoch milidetik
                parsed = pd.to_datetime(nums, unit="ms", utc=True, errors="coerce")
            else:                                      # epoch detik
                parsed = pd.to_datetime(nums, unit="s", utc=True, errors="coerce")
        else:
            parsed = pd.to_datetime(values, errors="coerce", utc=True)
        date_series = date_series.fillna(parsed)
        if date_series.notna().all():
            break
    out["date"] = date_series

    likes_col = resolved["likes"]
    out["likes"] = (pd.to_numeric(frame[likes_col], errors="coerce").fillna(0).astype(int)
                    if likes_col else 0)
    user_col = resolved["user"]
    out["user"] = (frame[user_col].fillna("unknown").astype(str).str.strip()
                   if user_col else "unknown")
    video_col = resolved["video_url"]
    out["video_url"] = (frame[video_col].fillna("").astype(str)
                        if video_col else "")

    out = out[out["text"].str.len() > 0].reset_index(drop=True)
    return out


def filter_dataset(df: pd.DataFrame, start: str, end: str) -> tuple[pd.DataFrame, dict]:
    """Saring rentang tanggal, buang kosong/pendek/duplikat per (user, text)."""
    stats = {"total": int(len(df))}
    if df.empty:
        stats.update(tanggal_gagal=0, dibuang_kosong=0, dibuang_pendek=0,
                     dibuang_duplikat=0, tersisa=0)
        return df.copy(), stats

    work = df.copy()
    work["user"] = work["user"].fillna("unknown").replace("", "unknown")

    bad_date = work["date"].isna()
    stats["tanggal_gagal"] = int(bad_date.sum())
    work = work[~bad_date]

    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    work = work[(work["date"] >= start_ts) & (work["date"] <= end_ts)]

    empty = work["text"].str.strip().eq("")
    stats["dibuang_kosong"] = int(empty.sum())
    work = work[~empty]

    short = work["text"].str.strip().str.len() < 3
    stats["dibuang_pendek"] = int(short.sum())
    work = work[~short]

    before = len(work)
    work = work.drop_duplicates(subset=["user", "text"], keep="first")
    stats["dibuang_duplikat"] = int(before - len(work))
    stats["tersisa"] = int(len(work))
    return work.reset_index(drop=True), stats


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------
_STEMMER = None
_STOPWORDS: set[str] | None = None


def get_stemmer():
    global _STEMMER
    if _STEMMER is None:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        _STEMMER = StemmerFactory().create_stemmer()
    return _STEMMER


def get_stopword_set() -> set[str]:
    global _STOPWORDS
    if _STOPWORDS is None:
        from Sastrawi.StopWordRemover.StopWordRemoverFactory import \
            StopWordRemoverFactory
        found = set(StopWordRemoverFactory().get_stop_words())
        _STOPWORDS = found - NEGATION_WORDS
    return _STOPWORDS


def normalize_text(text: str) -> str:
    """Bersihkan teks dan normalisasi slang. Belum membuang stopword."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    low = text.lower()

    for symbol, weight in EMOJI_MAP.items():
        if symbol in low:
            token = "emojipos" if weight > 0 else "emojineg"
            low = low.replace(symbol, f" {token} ")

    low = URL_RE.sub(" ", low)
    low = MENTION_RE.sub(" ", low)
    low = re.sub(r"\[\w+\]", " ", low)          # sisa placeholder [xxx]
    low = NON_ALPHA_RE.sub(" ", low)
    low = REPEAT_RE.sub(r"\1", low)

    tokens: list[str] = []
    for token in low.split():
        mapped = SLANG_DICT.get(token, token)
        tokens.extend(mapped.split())
    return " ".join(tokens)


def remove_stopwords(tokens: list[str]) -> list[str]:
    stops = get_stopword_set()
    return [t for t in tokens if t in NEGATION_WORDS or t not in stops]


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Tambah kolom norm_text (untuk label) dan clean_text (untuk TF-IDF)."""
    out = df.copy()
    out["norm_text"] = out["text"].map(normalize_text)
    stemmer = get_stemmer()

    def _clean(text: str) -> str:
        kept = remove_stopwords(text.split())
        return " ".join(stemmer.stem(t) for t in kept)

    out["clean_text"] = out["norm_text"].map(_clean)
    empty = out["clean_text"].str.strip().eq("")
    if empty.any():
        out.loc[empty, "clean_text"] = out.loc[empty, "norm_text"]
    return out


# --------------------------------------------------------------------------
# Leksikon & pelabelan hybrid
# --------------------------------------------------------------------------
def _read_weight_file(path: Path) -> Counter:
    weights: Counter = Counter()
    if not path.exists():
        return weights
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if i == 0 and line.lower().startswith("word"):
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 2:
            continue
        word, weight = parts[0].strip().lower(), parts[1].strip()
        try:
            weights[word] += int(float(weight))
        except ValueError:
            continue
    return weights


def load_inset(data_dir: Path = DATA_DIR) -> dict[str, float]:
    """Muat InSet dari salinan lokal; unduh hanya bila belum ada."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = {"pos": data_dir / "positive.tsv", "neg": data_dir / "negative.tsv"}
    if not all(p.exists() for p in paths.values()):
        for key, url in (("pos", f"{INSET_BASE}/positive.tsv"),
                         ("neg", f"{INSET_BASE}/negative.tsv")):
            try:
                response = requests.get(url, timeout=15)
                response.raise_for_status()
                paths[key].write_text(response.text, encoding="utf-8")
            except Exception as exc:                      # noqa: BLE001
                LOGGER.warning("Gagal mengunduh InSet %s: %s", key, exc)

    combined: Counter = Counter()
    for path in (paths["pos"], paths["neg"]):
        combined.update(_read_weight_file(path))
    return {w: float(v) for w, v in combined.items() if v != 0}


def load_domain_lexicon(path: Path = DATA_DIR / "domain_lexicon.tsv") -> dict[str, float]:
    path = Path(path)
    result: dict[str, float] = {}
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        try:
            result[parts[0].strip().lower()] = float(parts[1].strip())
        except ValueError:
            continue
    return result


def load_lexicon(data_dir: Path = DATA_DIR) -> dict[str, float]:
    """Gabungan InSet + kamus domain + token emoji; kata noise dibuang."""
    inset = load_inset(data_dir)
    domain = load_domain_lexicon(Path(data_dir) / "domain_lexicon.tsv")
    merged: dict[str, float] = dict(inset)
    merged.update(domain)
    merged.update({"emojipos": 1.0, "emojineg": -1.0})
    return {w: v for w, v in merged.items() if w not in NOISE_WORDS and v != 0}


def score_sentiment(text: str, lexicon: dict[str, float]) -> float:
    """Skor polaritas dengan flip negasi berjendela 3 token."""
    tokens = str(text or "").split()
    score = 0.0
    window = 0
    for token in tokens:
        if token in NEGATION_WORDS:
            window = 3
            continue
        value = lexicon.get(token, 0.0)
        if window > 0 and value > 0:
            value = -value
        score += value
        window = max(0, window - 1)
    return float(score)


def label_from_score(score: float, band: int = 1) -> str:
    if score > band:
        return "Positif"
    if score < -band:
        return "Negatif"
    return "Netral"


def build_dataset(df: pd.DataFrame, lexicon: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    if "norm_text" not in out.columns:
        out["norm_text"] = out["text"].map(normalize_text)
    out["score"] = out["norm_text"].map(lambda t: score_sentiment(t, lexicon))
    out["label_sentimen"] = out["score"].map(label_from_score)
    return out


# --------------------------------------------------------------------------
# Split & ekspor
# --------------------------------------------------------------------------
def split_dataset(df: pd.DataFrame, use_neutral: bool = True,
                  test_size: float = 0.2, random_state: int = 42
                  ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    from sklearn.model_selection import train_test_split

    work = df.copy().reset_index(drop=True)
    if "row_id" not in work.columns:
        work["row_id"] = range(len(work))
    if not use_neutral:
        work = work[work["label_sentimen"] != "Netral"]

    info = {"stratified": False, "catatan": ""}
    counts = work["label_sentimen"].value_counts()
    if len(counts) < 2:
        info["catatan"] = ("Hanya satu kelas tersedia; data tidak bisa dilatih. "
                           "Periksa hasil pelabelan.")
        return work, work.iloc[0:0], info

    small = [c for c in counts.index if work["label_sentimen"].eq(c).sum() < 2]
    if small:
        work = work[~work["label_sentimen"].isin(small)]
        counts = work["label_sentimen"].value_counts()
        info["catatan"] = f"Kelas dibuang karena < 2 baris: {', '.join(small)}. "

    if counts.min() >= 2:
        train, test = train_test_split(work, test_size=test_size,
                                       random_state=random_state,
                                       stratify=work["label_sentimen"])
        info["stratified"] = True
    else:
        train, test = train_test_split(work, test_size=test_size,
                                       random_state=random_state)
        info["catatan"] += ("stratify dilewati karena ada kelas dengan < 2 baris; "
                            "pembagian tidak seimbang per kelas. ")
    return train.reset_index(drop=True), test.reset_index(drop=True), info


EXPORT_COLUMNS = ["tanggal", "komentar_asli", "norm_text", "clean_text", "score",
                  "label_sentimen", "split", "likes", "user", "video_url"]


def export_datasets(df: pd.DataFrame, out_dir: Path = DATA_DIR,
                    sep: str = ",") -> dict[str, Path]:
    """Tulis tiga CSV (utf-8-sig) dan kembalikan path-nya."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = df.copy()
    if "date" in work.columns:
        work["tanggal"] = pd.to_datetime(work["date"], utc=True, errors="coerce") \
            .dt.strftime("%Y-%m-%d")
    elif "tanggal" in work.columns:
        work["tanggal"] = pd.to_datetime(work["tanggal"], errors="coerce") \
            .dt.strftime("%Y-%m-%d")
    if "text" in work.columns:
        work["komentar_asli"] = work["text"]
    for column in EXPORT_COLUMNS:
        if column not in work.columns:
            work[column] = ""

    paths = {
        "processed": out_dir / "dataset_processed_2026.csv",
        "train": out_dir / "dataset_train.csv",
        "test": out_dir / "dataset_test.csv",
    }
    work = work[EXPORT_COLUMNS]
    work.to_csv(paths["processed"], index=False, encoding="utf-8-sig", sep=sep)
    work[work["split"] == "latih"].to_csv(paths["train"], index=False,
                                          encoding="utf-8-sig", sep=sep)
    work[work["split"] == "uji"].to_csv(paths["test"], index=False,
                                        encoding="utf-8-sig", sep=sep)
    return paths


def run_pipeline(data_dir: Path = DATA_DIR, start: str = "2026-01-01",
                 end: str = "2026-11-18", use_neutral: bool = True,
                 raw_path: Path | None = None, text_col: str | None = None,
                 date_col: str | None = None, write: bool = True) -> dict:
    data_dir = Path(data_dir)
    source = Path(raw_path) if raw_path else find_raw_file(data_dir)
    raw = load_raw(source, text_col=text_col, date_col=date_col)
    filtered, stats = filter_dataset(raw, start, end)
    processed = preprocess_dataframe(filtered)
    lexicon = load_lexicon(data_dir)
    labeled = build_dataset(processed, lexicon)
    labeled = labeled.reset_index(drop=True)
    labeled["row_id"] = range(len(labeled))
    train, test, info = split_dataset(labeled, use_neutral=use_neutral)

    labeled["split"] = "latih"
    labeled.loc[labeled["row_id"].isin(test["row_id"]), "split"] = "uji"
    paths = export_datasets(labeled, data_dir) if write else {}

    summary = {
        "stats": stats,
        "split": info,
        "distribusi": labeled["label_sentimen"].value_counts().to_dict(),
        "train": len(train),
        "test": len(test),
        "paths": {k: str(v) for k, v in paths.items()},
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


# --------------------------------------------------------------------------
# Model & evaluasi
# --------------------------------------------------------------------------
def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(max_features=2500, ngram_range=(1, 2),
                           sublinear_tf=True)


def _metrics(y_true, y_pred, labels) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def train_and_evaluate(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, dict]:
    """Latih Naive Bayes & SVM. Vectorizer di-fit hanya pada data latih."""
    vectorizer = build_vectorizer()
    x_train = vectorizer.fit_transform(train["clean_text"].fillna("").astype(str))
    x_test = vectorizer.transform(test["clean_text"].fillna("").astype(str))
    y_train = train["label_sentimen"]
    y_test = test["label_sentimen"].reset_index(drop=True)
    labels = sorted(set(y_train) | set(y_test), key=lambda x: LABEL_ORDER.index(x)
                    if x in LABEL_ORDER else 99)

    models = {
        "Naive Bayes": MultinomialNB(alpha=0.1),
        "SVM": LinearSVC(class_weight="balanced"),
    }
    results: dict[str, dict] = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        results[name] = {
            "model": model,
            "vectorizer": vectorizer,
            "metrics": _metrics(y_test, y_pred, labels),
            "report": classification_report(y_test, y_pred, labels=labels,
                                            output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels).tolist(),
            "labels": labels,
            "y_pred": list(y_pred),
            "prediksi": pd.Series(y_pred, index=test.index),
        }
    return results


def pick_best(results: dict) -> str:
    return max(results, key=lambda name: results[name]["metrics"]["f1_macro"])


def two_proportion_ztest(c1: int, n1: int, c2: int, n2: int) -> tuple[float, float]:
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0
    p1, p2 = c1 / n1, c2 / n2
    pooled = (c1 + c2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    p = math.erfc(abs(z) / math.sqrt(2))
    return float(z), float(p)


def period_comparison(df: pd.DataFrame) -> dict:
    """Bandingkan sentimen Jan-Jun vs Sep-Nov; Jul-Agu dikeluarkan."""
    work = df.copy()
    col = "date" if "date" in work.columns else "tanggal"
    work[col] = pd.to_datetime(work[col], utc=True, errors="coerce")
    months = work[col].dt.strftime("%Y-%m")

    awal = work[months.between("2026-01", "2026-06")]
    akhir = work[months.between("2026-09", "2026-11")]
    excluded = int(months.between("2026-07", "2026-08").sum())

    def _summary(part: pd.DataFrame) -> dict:
        n = len(part)
        counts = part["label_sentimen"].value_counts()
        return {
            "n": int(n),
            "Positif": int(counts.get("Positif", 0)),
            "Negatif": int(counts.get("Negatif", 0)),
            "Netral": int(counts.get("Netral", 0)),
            "persen_pos": float(counts.get("Positif", 0) / n) if n else 0.0,
            "persen_neg": float(counts.get("Negatif", 0) / n) if n else 0.0,
            "persen_net": float(counts.get("Netral", 0) / n) if n else 0.0,
        }

    a, b = _summary(awal), _summary(akhir)
    _, p_pos = two_proportion_ztest(a["Positif"], a["n"], b["Positif"], b["n"])
    _, p_neg = two_proportion_ztest(a["Negatif"], a["n"], b["Negatif"], b["n"])
    return {"awal": a, "akhir": b, "dikecualikan": excluded,
            "p_pos": float(p_pos), "p_neg": float(p_neg)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline sentimen MBG Blitar")
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end", default="2026-11-18")
    parser.add_argument("--biner", action="store_true",
                        help="Buang kelas Netral (klasifikasi biner)")
    args = parser.parse_args()
    run_pipeline(start=args.start, end=args.end, use_neutral=not args.biner)


if __name__ == "__main__":
    main()
