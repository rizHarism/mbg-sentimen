import json
from pathlib import Path

import pandas as pd
import pytest

from text_mining_pipeline import (
    build_dataset, export_datasets, filter_dataset, label_from_score,
    load_lexicon, load_raw, normalize_text, period_comparison, pick_best,
    remove_stopwords, score_sentiment, split_dataset, train_and_evaluate,
    two_proportion_ztest,
)

LEX = {"enak": 3, "bagus": 2, "basi": -4, "keracunan": -5, "emojipos": 2}


def _frame(texts, dates=("2026-02-01",), users=None):
    n = len(texts)
    return pd.DataFrame({
        "text": texts,
        "date": pd.to_datetime([dates[i % len(dates)] for i in range(n)], utc=True),
        "likes": [0] * n,
        "user": users or [f"u{i}" for i in range(n)],
        "video_url": ["v"] * n,
    })


# --- ingest ---------------------------------------------------------------
def test_load_raw_json_apify(tmp_path: Path):
    p = tmp_path / "raw.json"
    p.write_text(json.dumps([
        {"text": "enak banget", "createTimeISO": "2026-02-01T10:00:00Z",
         "diggCount": 3, "uniqueId": "budi", "videoWebUrl": "v1"},
    ]), encoding="utf-8")
    df = load_raw(p)
    assert df["text"].tolist() == ["enak banget"]
    assert df["likes"].tolist() == [3]
    assert df["user"].tolist() == ["budi"]


def test_load_raw_epoch_milidetik(tmp_path: Path):
    p = tmp_path / "raw.json"
    p.write_text(json.dumps([{"text": "halo", "createTime": 1770000000000}]), encoding="utf-8")
    df = load_raw(p)
    assert df["date"].notna().all() and df.loc[0, "date"].year == 2026


def test_load_raw_semua_teks_kosong(tmp_path: Path):
    p = tmp_path / "raw.json"
    p.write_text(json.dumps([{"text": "", "createTimeISO": "2026-02-01T00:00:00Z"}]),
                 encoding="utf-8")
    assert load_raw(p).empty


def test_load_raw_tanpa_kolom_teks_beri_pesan(tmp_path: Path):
    p = tmp_path / "raw.json"
    p.write_text(json.dumps([{"createTimeISO": "2026-02-01T00:00:00Z"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="Kolom teks"):
        load_raw(p)


def test_filter_dedup_per_user_dan_buang_pendek():
    df = _frame(["req sate min", "req sate min", "req sate min", "ok", "porsi kecil"],
                users=["a", "b", "a", "c", "d"])
    out, stats = filter_dataset(df, "2026-01-01", "2026-11-18")
    assert len(out) == 3
    assert stats["dibuang_duplikat"] == 1
    assert stats["dibuang_pendek"] == 1


def test_filter_user_kosong_tidak_semua_terbuang():
    df = _frame(["satu", "dua", "tiga"])
    df["user"] = [None, None, "x"]
    out, _ = filter_dataset(df, "2026-01-01", "2026-11-18")
    assert len(out) == 3


# --- preprocessing -------------------------------------------------------
def test_normalize_buang_url_mention_angka_dan_repeat():
    out = normalize_text("cek https://t.co/x @sppg #mbg 2026 mantappp")
    assert "http" not in out and "@" not in out and "2026" not in out
    assert "mantap" in out


def test_normalize_emoji_dan_slang():
    assert "emojipos" in normalize_text("enak banget [Terkekeh]")
    assert "emojineg" in normalize_text("basi [Sedih]")
    assert normalize_text("ora enak, gak apik") == "tidak enak tidak bagus"


def test_negasi_bertahan_setelah_stopword():
    toks = remove_stopwords(["saya", "tidak", "suka", "yang", "basi"])
    assert "tidak" in toks and "basi" in toks
    assert "saya" not in toks and "yang" not in toks


# --- leksikon ------------------------------------------------------------
def _write_lexicon(dirpath: Path):
    (dirpath / "positive.tsv").write_text("word\tweight\nenak\t5\nmantap\t3\n", encoding="utf-8")
    (dirpath / "negative.tsv").write_text("word\tweight\nenak\t-4\nyang\t-5\n", encoding="utf-8")
    (dirpath / "domain_lexicon.tsv").write_text("enak\t4\nkeracunan\t-5\n", encoding="utf-8")


def test_lexicon_jumlah_domain_menimpa_noise_dibuang(tmp_path: Path):
    _write_lexicon(tmp_path)
    lex = load_lexicon(tmp_path)
    assert lex["enak"] == 4          # domain menimpa
    assert "yang" not in lex         # noise dibuang
    assert lex["keracunan"] == -5
    assert lex["emojipos"] == 1


def test_lexicon_offline_tetap_pakai_lokal(tmp_path: Path, monkeypatch):
    _write_lexicon(tmp_path)
    def boom(*a, **k):
        raise OSError("offline")
    monkeypatch.setattr("text_mining_pipeline.requests.get", boom)
    lex = load_lexicon(tmp_path)
    assert lex["mantap"] == 3 and lex["enak"] == 4


def test_flip_negasi_dan_ambang_netral():
    assert score_sentiment("tidak enak", LEX) < 0
    assert score_sentiment("enak bagus", LEX) > 0
    assert label_from_score(1) == "Netral"
    assert label_from_score(-1) == "Netral"
    assert label_from_score(2) == "Positif"
    assert label_from_score(-2) == "Negatif"


def test_keracunan_dihitung_negatif():
    assert score_sentiment("anak keracunan", LEX) < -3


def test_build_dataset():
    df = pd.DataFrame({"text": ["enak", "basi", "req sate"],
                       "norm_text": ["enak", "basi", "req sate"]})
    out = build_dataset(df, LEX)
    assert out["label_sentimen"].tolist() == ["Positif", "Negatif", "Netral"]


# --- split & ekspor ------------------------------------------------------
def test_split_80_20_dan_tidak_tumpang_tindih():
    df = build_dataset(_frame(["enak"] * 30 + ["basi"] * 30 + ["req"] * 30), LEX)
    tr, te, info = split_dataset(df)
    assert len(tr) + len(te) == 90
    assert abs(len(te) / 90 - 0.2) < 0.04
    assert info["stratified"] is True
    assert set(tr["row_id"]).isdisjoint(set(te["row_id"]))


def test_kelas_terlalu_kecil_dibuang_dengan_catatan():
    df = build_dataset(_frame(["enak"] * 40 + ["basi"] * 39 + ["req"]), LEX)
    tr, te, info = split_dataset(df)
    assert info["stratified"] is True          # kelas 1 baris dibuang dulu
    assert "Netral" not in set(tr["label_sentimen"])
    assert "dibuang" in info["catatan"].lower()
    tr, te, _ = split_dataset(df, use_neutral=False)
    assert "Netral" not in set(tr["label_sentimen"])


def test_ekspor_utf8_sig(tmp_path: Path):
    df = build_dataset(_frame(["enak", "basi"]), LEX)
    df["split"] = "latih"
    paths = export_datasets(df, tmp_path)
    assert paths["processed"].read_bytes().startswith(b"\xef\xbb\xbf")
    assert set(paths) == {"processed", "train", "test"}


def test_ekspor_pemisah_titik_koma(tmp_path: Path):
    df = build_dataset(_frame(["enak, mantap", "basi"]), LEX)
    df["split"] = "latih"
    paths = export_datasets(df, tmp_path, sep=";")
    teks = paths["processed"].read_text(encoding="utf-8-sig")
    assert teks.splitlines()[0].count(";") >= 8
    back = pd.read_csv(paths["processed"], encoding="utf-8-sig", sep=";")
    assert "enak, mantap" in back["komentar_asli"].tolist()


# --- model & uji periode -------------------------------------------------
def test_model_terlatih_dan_vectorizer_tidak_bocor():
    df = build_dataset(_frame(
        ["enak bagus porsi"] * 40 + ["basi keracunan anak"] * 40), LEX)
    df["clean_text"] = df["norm_text"]
    tr, te, _ = split_dataset(df, use_neutral=False)
    res = train_and_evaluate(tr, te)
    assert set(res) == {"Naive Bayes", "SVM"}
    assert res["SVM"]["metrics"]["accuracy"] > 0.8
    assert len(res["SVM"]["prediksi"]) == len(te)
    assert res["SVM"]["vectorizer"].vocabulary_.get("keracunan") is not None


def test_pick_best_dan_ztest():
    res = {"Naive Bayes": {"metrics": {"f1_macro": 0.7}},
           "SVM": {"metrics": {"f1_macro": 0.8}}}
    assert pick_best(res) == "SVM"
    z, p = two_proportion_ztest(114, 531, 538, 2332)
    assert abs(z + 0.79) < 0.02 and abs(p - 0.427) < 0.02


def test_period_comparison_mengeluarkan_jul_agu():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2026-02-01"] * 10 + ["2026-07-15"] * 4
                               + ["2026-10-01"] * 10, utc=True),
        "label_sentimen": ["Positif"] * 20 + ["Negatif"] * 4,
    })
    out = period_comparison(df)
    assert out["awal"]["n"] == 10 and out["akhir"]["n"] == 10
    assert out["dikecualikan"] == 4

