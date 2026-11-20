"""Dashboard Streamlit: Analisis Sentimen MBG Blitar 2026."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from wordcloud import WordCloud

from text_mining_pipeline import (period_comparison, pick_best, run_pipeline,
                                  train_and_evaluate)
from sklearn.feature_extraction.text import CountVectorizer

st.set_page_config(page_title="Sentimen MBG Blitar 2026", layout="wide")

DATA = Path("data")
PROCESSED = DATA / "dataset_processed_2026.csv"
TRAIN = DATA / "dataset_train.csv"
TEST = DATA / "dataset_test.csv"
VALIDATION = DATA / "validasi_manual.csv"

COLORS = {"Positif": "#2E9E5B", "Negatif": "#D64545", "Netral": "#9AA0A6"}


@st.cache_data(show_spinner=False)
def load_processed(path: str, mtime: float) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce")
    for col in ("komentar_asli", "clean_text", "norm_text", "label_sentimen"):
        if col in df.columns:
            df[col] = df[col].fillna("")
    return df


@st.cache_data(show_spinner=False)
def load_csv(path: str, mtime: float) -> pd.DataFrame:
    """Baca CSV dengan pemisah koma; jatuh ke titik koma (Excel Indonesia)."""
    for sep in (",", ";"):
        try:
            return pd.read_csv(path, encoding="utf-8-sig", sep=sep)
        except pd.errors.ParserError:
            continue
    return pd.read_csv(path, encoding="utf-8-sig", sep=None, engine="python")


def file_mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner="Melatih model...")
def evaluate_cached(train_path: str, train_mtime: float,
                    test_path: str, test_mtime: float) -> dict:
    """Latih sekali, cache hasilnya (tanpa objek model agar bisa di-pickle)."""
    train = pd.read_csv(train_path, encoding="utf-8-sig")
    test = pd.read_csv(test_path, encoding="utf-8-sig")
    train = train.rename(columns={"label_sentimen": "label_sentimen"})
    results = train_and_evaluate(train, test)
    return {
        name: {k: v for k, v in res.items() if k not in ("model", "vectorizer")}
        for name, res in results.items()
    }


# ---------------------------------------------------------------- sidebar
st.sidebar.title("Kontrol Data")

uploaded = st.sidebar.file_uploader("Unggah berkas baru (.csv / .json)",
                                    type=["csv", "json"])
text_col = date_col = None
if uploaded is not None:
    try:
        preview = (pd.read_json(io.BytesIO(uploaded.getvalue()))
                   if uploaded.name.endswith(".json")
                   else pd.read_csv(io.BytesIO(uploaded.getvalue())))
        options = list(preview.columns)
        text_col = st.sidebar.selectbox("Kolom komentar", options)
        date_col = st.sidebar.selectbox(
            "Kolom tanggal", options,
            index=options.index(next((c for c in options if "time" in c.lower()
                                      or "tanggal" in c.lower()), options[0])))
        if st.sidebar.button("Proses berkas ini"):
            target = DATA / f"upload_{uploaded.name}"
            target.write_bytes(uploaded.getvalue())
            with st.spinner("Menjalankan pipeline..."):
                run_pipeline(raw_path=target, text_col=text_col,
                             date_col=date_col)
            st.cache_data.clear()
            st.rerun()
    except Exception as exc:                              # noqa: BLE001
        st.sidebar.error(f"Berkas tidak bisa dibaca: {exc}")

if not PROCESSED.exists():
    st.title("Belum ada data hasil olahan")
    st.info("Jalankan `python text_mining_pipeline.py` lalu muat ulang halaman ini.")
    st.stop()

df_all = load_processed(str(PROCESSED), file_mtime(PROCESSED))

start_default = pd.Timestamp("2026-01-01").date()
end_default = pd.Timestamp("2026-11-18").date()
date_range = st.sidebar.date_input("Rentang tanggal", (start_default, end_default))
with_neutral = st.sidebar.checkbox("Sertakan kelas Netral", value=True)
keyword = st.sidebar.text_input("Filter keyword lokal", placeholder="telur, keracunan")
max_per_video = st.sidebar.slider("Maks komentar per video (tampilan)", 50, 800, 400,
                                  step=50)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_ts, end_ts = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
else:
    start_ts, end_ts = pd.Timestamp(start_default), pd.Timestamp(end_default)

view = df_all[(df_all["tanggal"] >= start_ts) & (df_all["tanggal"] <= end_ts)]
if not with_neutral:
    view = view[view["label_sentimen"] != "Netral"]
if keyword:
    mask = view["komentar_asli"].str.contains(keyword, case=False, na=False)
    view = view[mask]
if "video_url" in view.columns and view["video_url"].ne("").any():
    view = (view.sort_values("likes", ascending=False)
                .groupby("video_url").head(max_per_video)
                .reset_index(drop=True))
view = view.reset_index(drop=True)

st.title("Analisis Sentimen MBG Blitar 2026")
st.caption(f"{len(view):,} komentar valid ditampilkan · sumber: 15 video TikTok · "
           "November 2026 bersifat parsial")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Ringkasan & Waktu", "Evaluasi Model", "Kata & Topik", "Eksplorasi & Unduh"])

# ------------------------------------------------------------------- tab 1
with tab1:
    total = len(view)
    share = (view["label_sentimen"].value_counts(normalize=True) * 100) if total else pd.Series(dtype=float)
    cols = st.columns(4)
    cols[0].metric("Total Komentar Valid", f"{total:,}")
    for col, label in zip(cols[1:], ["Positif", "Negatif", "Netral"]):
        col.metric(f"{label}", f"{share.get(label, 0):.1f}%")

    if total:
        monthly = (view.assign(bulan=view["tanggal"].dt.strftime("%Y-%m"))
                       .groupby(["bulan", "label_sentimen"]).size()
                       .unstack(fill_value=0))
        monthly["n"] = monthly.sum(axis=1)

        fig = go.Figure()
        fig.add_bar(x=monthly.index, y=monthly["n"], name="Volume komentar",
                    marker_color="#D9D9D9", yaxis="y2", opacity=0.6)
        for label in ["Positif", "Negatif", "Netral"]:
            if label in monthly:
                pct = (monthly[label] / monthly["n"] * 100)
                fig.add_scatter(x=monthly.index, y=pct, name=label, mode="lines+markers",
                                line=dict(color=COLORS[label], width=2))
        fig.update_layout(
            title="Distribusi sentimen per bulan (garis = %, batang = jumlah komentar)",
            yaxis=dict(title="Persentase (%)", range=[0, 100]),
            yaxis2=dict(title="Jumlah", overlaying="y", side="right", showgrid=False),
            legend=dict(orientation="h"), hovermode="x unified", height=420)
        fig.add_annotation(text="Mar–Agu tanpa video baru · Nov 2026 parsial",
                           xref="paper", yref="paper", x=0.02, y=1.08, showarrow=False,
                           font=dict(size=11, color="#777"))
        st.plotly_chart(fig, use_container_width=True)

        left, right = st.columns([1, 1])
        with left:
            donut = px.pie(view, names="label_sentimen", hole=0.55,
                           color="label_sentimen", color_discrete_map=COLORS,
                           title="Proporsi keseluruhan")
            st.plotly_chart(donut, use_container_width=True)
        with right:
            per = period_comparison(view)
            st.subheader("Awal (Jan–Jun) vs Akhir (Sep–Nov)")
            st.dataframe(pd.DataFrame([
                {"Periode": "Awal 2026", "n": per["awal"]["n"],
                 "Positif": f"{per['awal']['persen_pos']:.1%}",
                 "Negatif": f"{per['awal']['persen_neg']:.1%}",
                 "Netral": f"{per['awal']['persen_net']:.1%}"},
                {"Periode": "Akhir 2026", "n": per["akhir"]["n"],
                 "Positif": f"{per['akhir']['persen_pos']:.1%}",
                 "Negatif": f"{per['akhir']['persen_neg']:.1%}",
                 "Netral": f"{per['akhir']['persen_net']:.1%}"},
            ]), use_container_width=True, hide_index=True)
            verdict = ("berbeda nyata" if per["p_neg"] < 0.05 else "tidak berbeda nyata")
            st.write(f"Uji proporsi Negatif: p = {per['p_neg']:.4f} → **{verdict}**. "
                     f"Proporsi Positif: p = {per['p_pos']:.4f}.")
            st.caption(f"Jul–Agu ({per['dikecualikan']} komentar) dikeluarkan dari uji "
                       "karena tidak ada video yang terbit pada periode itu.")

# ------------------------------------------------------------------- tab 2
with tab2:
    if TRAIN.exists() and TEST.exists():
        train = load_csv(str(TRAIN), file_mtime(TRAIN))
        test = load_csv(str(TEST), file_mtime(TEST))
        if len(train) and len(test) and train["label_sentimen"].nunique() > 1:
            results = evaluate_cached(str(TRAIN), file_mtime(TRAIN),
                                      str(TEST), file_mtime(TEST))
            metric_rows = []
            for name, res in results.items():
                m = res["metrics"]
                metric_rows.append({
                    "Model": name,
                    "Accuracy": round(m["accuracy"], 3),
                    "Precision (macro)": round(m["precision_macro"], 3),
                    "Recall (macro)": round(m["recall_macro"], 3),
                    "F1 (macro)": round(m["f1_macro"], 3),
                })
            st.dataframe(pd.DataFrame(metric_rows), use_container_width=True,
                         hide_index=True)

            cols = st.columns(2)
            for col, (name, res) in zip(cols, results.items()):
                cm = pd.DataFrame(res["confusion_matrix"], index=res["labels"],
                                  columns=res["labels"])
                fig_cm = px.imshow(cm, text_auto=True, aspect="auto",
                                   color_continuous_scale="Blues",
                                   title=f"Confusion Matrix — {name}")
                col.plotly_chart(fig_cm, use_container_width=True)

            best = pick_best(results)
            st.success(f"Model dengan F1-macro terbaik: **{best}**. "
                       "Accuracy tidak dipakai sebagai penentu karena kelas tidak seimbang.")
            st.caption("Metrik dihitung pada dataset yang diekspor (3 kelas). "
                       "Untuk model biner jalankan `python text_mining_pipeline.py --biner`.")

            if VALIDATION.exists():
                val = load_csv(str(VALIDATION), file_mtime(VALIDATION))
                val = val.loc[:, [c for c in val.columns if not c.startswith("Unnamed")]]
                st.subheader("Validasi pelabelan")
                if "tinjauan_manual" not in val.columns:
                    st.info("Kolom `tinjauan_manual` tidak ditemukan pada berkas validasi.")
                    val = None
            else:
                val = None

            if VALIDATION.exists() and val is not None:
                manual = (val["tinjauan_manual"].fillna("").astype(str)
                          .str.strip().str.lower())
                reviewed = int((manual != "").sum())
                total = len(val)

                # Manusia boleh menulis label langsung (negatif/netral/positif)
                # atau "sesuai" / "tidak".
                label_map = {"negatif": "Negatif", "netral": "Netral",
                             "positif": "Positif"}
                manual_label = manual.map(label_map)
                judged = manual_label.notna()
                agree_ai = (val["label_ai"] == val["label_hybrid"]).mean()

                st.write(f"Peninjauan manusia: **{reviewed} dari {total}** baris sampel.")
                cols = st.columns(3)
                cols[0].metric("Hybrid vs label AI", f"{agree_ai:.1%}")
                if judged.any():
                    cols[1].metric(
                        "Hybrid vs tinjauan manusia",
                        f"{(manual_label[judged] == val.loc[judged, 'label_hybrid']).mean():.1%}",
                        help=f"dihitung pada {int(judged.sum())} baris yang sudah dinilai")
                    cols[2].metric(
                        "Label AI vs tinjauan manusia",
                        f"{(manual_label[judged] == val.loc[judged, 'label_ai']).mean():.1%}",
                        help="mengukur seberapa dekat pembanding AI dengan penilaian manusia")
                else:
                    cols[1].metric("Hybrid vs tinjauan manusia", "—",
                                   help="isi kolom tinjauan_manual dengan "
                                        "negatif / netral / positif")
                    cols[2].metric("Label AI vs tinjauan manusia", "—")

                if reviewed == 0:
                    st.info(
                        "Kolom `tinjauan_manual` pada `data/validasi_manual.csv` masih "
                        "kosong. Isi minimal 20 baris dengan `negatif`, `netral`, atau "
                        "`positif` agar angka tinjauan manusia muncul.")
                else:
                    st.caption(
                        "Label AI dibuat oleh model bahasa, bukan manusia; angka "
                        "'Hybrid vs label AI' mengukur konsistensi aturan, bukan akurasi. "
                        "Angka 'vs tinjauan manusia' adalah penilaian Anda sendiri dan "
                        "boleh dikutip sebagai validasi manual.")
                if reviewed and (manual == "sesuai").any():
                    st.caption("Sebagian baris memakai format `sesuai`/`tidak`. "
                               "Format itu hanya dihitung sebagai 'sudah ditinjau' dan "
                               "tidak ikut perhitungan kesesuaian label.")
        else:
            st.warning("Data latih/uji belum cukup untuk melatih model.")
    else:
        st.warning("dataset_train.csv / dataset_test.csv belum ada.")

# ------------------------------------------------------------------- tab 3
with tab3:
    if len(view):
        pos_text = " ".join(view.loc[view["label_sentimen"] == "Positif", "clean_text"].dropna())
        neg_text = " ".join(view.loc[view["label_sentimen"] == "Negatif", "clean_text"].dropna())
        cols = st.columns(2)
        for col, (label, text, cmap) in zip(cols, [
                ("Positif", pos_text, "Greens"), ("Negatif", neg_text, "Reds")]):
            with col:
                st.subheader(f"WordCloud — {label}")
                if text.strip():
                    cloud = WordCloud(width=700, height=380, background_color="white",
                                      colormap=cmap, random_state=42).generate(text)
                    st.image(cloud.to_array(), use_container_width=True)
                else:
                    st.info("Tidak ada komentar pada kelas ini.")

        st.subheader("Contoh komentar per kelas")
        ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 2])
        n_examples = ctrl1.slider("Jumlah contoh per kelas", 3, 10, 5)
        mode = ctrl2.selectbox("Cara memilih contoh", ["Acak", "Skor ekstrem"])
        hide_empty = ctrl3.checkbox("Sembunyikan yang clean_text kosong", value=True)

        if "contoh_seed" not in st.session_state:
            st.session_state.contoh_seed = 42
        if st.button("Acak ulang contoh"):
            st.session_state.contoh_seed += 1
        seed = st.session_state.contoh_seed

        pool = view.copy()
        if hide_empty:
            pool = pool[pool["clean_text"].fillna("").str.strip().ne("")]

        cols = st.columns(2)
        for col, label in zip(cols, ["Positif", "Negatif"]):
            part = pool[pool["label_sentimen"] == label]
            with col:
                st.markdown(f"**{label}** — {len(part):,} komentar tersedia")
                if part.empty:
                    st.info("Tidak ada komentar pada kelas ini untuk filter aktif.")
                    continue
                if mode == "Skor ekstrem":
                    chosen = (part.nlargest(n_examples, "score") if label == "Positif"
                              else part.nsmallest(n_examples, "score"))
                else:
                    chosen = part.sample(min(n_examples, len(part)), random_state=seed)
                st.dataframe(
                    chosen[["komentar_asli", "norm_text", "clean_text",
                            "score", "likes", "tanggal"]],
                    use_container_width=True, hide_index=True)
        st.caption("Kutip `komentar_asli` di laporan; `norm_text` dan `clean_text` "
                   "hanya untuk menunjukkan apa yang dibaca model.")

        st.subheader("10 frasa paling sering")
        vec = CountVectorizer(ngram_range=(1, 2), max_features=10)
        try:
            matrix = vec.fit_transform(view["clean_text"].fillna(""))
            freq = pd.DataFrame({"frasa": vec.get_feature_names_out(),
                                 "jumlah": matrix.sum(axis=0).tolist()[0]})
            st.dataframe(freq.sort_values("jumlah", ascending=False),
                         use_container_width=True, hide_index=True)
        except ValueError:
            st.info("Teks terlalu pendek untuk analisis frasa.")

        st.subheader("Transformasi dialek & bahasa gaul")
        samples = view[view["norm_text"] != view["komentar_asli"].str.lower()]
        st.dataframe(samples[["komentar_asli", "norm_text"]].head(10),
                     use_container_width=True, hide_index=True)
        st.caption("Kosakata Blitaran murni sangat sedikit di data ini; mayoritas "
                   "normalisasi berasal dari bahasa gaul umum dan Jawa Timuran.")

        st.subheader("Komentar paling banyak disukai")
        st.dataframe(view.nlargest(10, "likes")[["komentar_asli", "likes",
                                                 "label_sentimen"]],
                     use_container_width=True, hide_index=True)

# ------------------------------------------------------------------- tab 4
with tab4:
    table = view.copy()
    table["Prediksi Model"] = "-"
    if TRAIN.exists() and TEST.exists():
        try:
            results = evaluate_cached(str(TRAIN), file_mtime(TRAIN),
                                      str(TEST), file_mtime(TEST))
            best = pick_best(results)
            pred = results[best]["prediksi"]
            test_view = table[table["split"] == "uji"].reset_index(drop=True)
            if len(test_view) == len(pred):
                table.loc[table["split"] == "uji", "Prediksi Model"] = pred.tolist()
        except Exception:                                  # noqa: BLE001
            pass

    st.caption("Kolom 'Prediksi Model' hanya terisi pada baris data uji; "
               "baris latih sengaja dibiarkan '-' agar tidak menampilkan prediksi in-sample.")
    search = st.text_input("Cari komentar")
    if search:
        table = table[table["komentar_asli"].str.contains(search, case=False, na=False)]
    st.dataframe(table[["tanggal", "komentar_asli", "clean_text",
                        "label_sentimen", "Prediksi Model"]],
                 use_container_width=True, hide_index=True, height=420)

    sep_choice = st.checkbox("Pakai pemisah titik koma ( ; ) untuk Excel Indonesia",
                             value=False)
    sep = ";" if sep_choice else ","
    st.caption("Encoding UTF-8 dengan BOM. Bila Excel menumpuk semua kolom jadi satu, "
               "buka lewat Data → From Text/CSV → pilih pemisah yang sesuai.")

    sources: list[tuple[str, pd.DataFrame | None]] = [
        ("dataset_processed_2026.csv", df_all),
        ("dataset_train.csv", load_csv(str(TRAIN), file_mtime(TRAIN)) if TRAIN.exists() else None),
        ("dataset_test.csv", load_csv(str(TEST), file_mtime(TEST)) if TEST.exists() else None),
        ("validasi_manual.csv", load_csv(str(VALIDATION), file_mtime(VALIDATION))
         if VALIDATION.exists() else None),
    ]
    cols = st.columns(4)
    for col, (name, frame) in zip(cols, sources):
        if frame is not None:
            payload = frame.to_csv(index=False, sep=sep).encode("utf-8-sig")
            col.download_button(name, payload, file_name=name,
                                use_container_width=True)
