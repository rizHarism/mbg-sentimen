# Referensi Laporan & PPT — Analisis Sentimen MBG Blitar 2026

Dokumen ini bahan acuan untuk kelompok. Semua angka di sini **hasil hitungan nyata**
dari aplikasi, bukan perkiraan. Bila ada yang diubah pada data, hitung ulang lewat
`python text_mining_pipeline.py`.

---

## 1. Identitas penelitian

| Butir | Isi |
|---|---|
| Judul | Analisis Sentimen Masyarakat terhadap Program Makan Bergizi Gratis Sepanjang Tahun 2026 Berdasarkan Opini Warganet Blitar di TikTok |
| Objek | Komentar TikTok pada video berkonteks MBG wilayah Blitar |
| Metode | Text mining: preprocessing bahasa Indonesia, pelabelan leksikon, TF-IDF, Naive Bayes vs SVM |
| Luaran | Pipeline Python + dashboard Streamlit + 3 dataset CSV |

**Rumusan masalah yang bisa dipakai:**

1. Bagaimana distribusi sentimen komentar warganet terhadap MBG Blitar sepanjang 2026?
2. Bagaimana performa Naive Bayes dibanding SVM pada klasifikasi sentimen tersebut?
3. Apakah proporsi sentimen berbeda antara paruh awal dan paruh akhir 2026?

---

## 2. Pencarian sumber data

**Alat:** Apify — aktor **`clockworks/tiktok-comments-scraper`**, dijalankan manual
lewat Apify Console (bukan lewat API).

**Bukti pengambilan data** disimpan di folder `apify/` dan dipakai sebagai lampiran
bab metodologi:

| Berkas | Isi yang bisa dikutip |
|---|---|
| `Screenshot 2026-11-20 145458.png` | Halaman *Input* aktor: 15 URL sumber, peringatan kuota gratis habis |
| `Screenshot 2026-11-20 145521.png` | Batas crawl: maks **2.000** komentar/posting, **200** komentar top-level/posting, **100** balasan/komentar |
| `Screenshot 2026-11-20 145753.png` | Tabel *Output*: 9 kolom mentah, contoh unggahan SPPG Mojorejo |
| `Screenshot 2026-11-20 145802.png` | Tabel *Output*: contoh unggahan `storymasibin` |
| `Screenshot 2026-11-20 145850.png` | Dialog *Export dataset*: format CSV, **3.094 baris**, run selesai 5 m 30 s, biaya **$3,868** |
| `Screenshot 2026-11-20 145554.png`, `…145815.png` | Pendukung proses crawl |
| `source-video.txt` | 15 URL sumber + kata kunci pencariannya |

**Batas 200 komentar top-level per unggahan** penting disebut: data ini bukan
himpunan lengkap, melainkan 200 komentar teratas per unggahan ditambah balasannya.
Unggahan terbesar hanya menyumbang 581 baris karena batas itu, bukan karena
komentarnya habis.

**Kuota habis.** Peringatan "exceeded the free limit of $5" pada tangkapan layar
menjelaskan mengapa data tidak bisa ditambah lagi. Kalau dosen bertanya "kenapa
tidak ditambah videonya?", inilah jawabannya — bukan karena data dianggap cukup.

**Satu detail yang mungkin ditanya.** Log run pada tangkapan layar
`…145850.png` mencatat 1.339 komentar untuk run terakhir, sedangkan dataset
berisi 3.094 baris. Halaman aktor menunjukkan **Runs: 2**, jadi dataset menampung
gabungan dua run. Sebutkan apa adanya bila ditanya; jangan mengklaim 3.094
berasal dari satu kali proses.

**Kata kunci pencarian video** (tersimpan di `apify/source-video.txt`): `mbg blitar
hari ini`, `berita mbg blitar`, `mbg blitar`.

**Cara memilih video:** bukan video terpopuler, tetapi video yang relevan dengan MBG
di Blitar, mencakup akun resmi SPPG, media lokal, dan akun pribadi.

**15 video dari 14 akun** (dua di antaranya unggahan foto, bukan video, tetapi
komentarnya tetap terambil): `sppg.tlogo.2.kani` (2 unggahan), `sppgtalunblitar_`,
`sppgkandangan01blitar`, `sppg.tegalrejo.bl`, `sppg.kuningan.kanigoro`,
`sppgmojorejo2`, `sppgblitarponggok05`, `storymasibin`, `portaljtvcom`,
`surya.co.id`, `mayangkaranews`, `officiallambeturah2`, `memo.co.id`,
`kantor_pengacara`.

**Volume:**

- 3.094 baris mentah; 49 di antaranya teks kosong sehingga 3.045 komentar berisi.
- 2.556 komentator unik.
- Rentang waktu: November 2025 – 18 November 2026.
- Kolom tersedia: `text`, `diggCount` (likes), `replyCommentTotal`,
  `createTimeISO`, `uniqueId`, `videoWebUrl`, `uid`, `cid`, `avatarThumbnail`.

**Bulan publikasi tiap unggahan** (diturunkan dari ID unggahan: 32 bit atas ID
TikTok adalah epoch Unix) menentukan sebaran komentar:

| Bulan publikasi | Jumlah unggahan | Komentar yang masuk |
|---|---|---|
| Nov 2025 | 1 | 237 |
| Des 2025 | 1 | 150 |
| Jan 2026 | 1 | 82 |
| Feb 2026 | 2 | 301 |
| **Mar–Agu 2026** | **0** | **0** |
| Sep 2026 | 3 | 712 |
| Okt 2026 | 3 | 1.452 |
| Nov 2026 | 4 | 111 |

**Ini temuan penting, bukan kebetulan:** tidak ada unggahan yang terbit
Maret–Agustus 2026, sehingga komentar pada bulan-bulan itu hampir kosong
(Jul 7, Agu 3 baris). Kurva bulanan mencerminkan jadwal publikasi, **bukan**
perubahan opini masyarakat. Tiga unggahan terbesar (`liismrn` 581, `nupi_kecil22`
529, `storymasibin` 461) menyumbang 51,6% data.

---

## 3. Batasan data (wajib ditulis di bab metodologi)

1. **Tidak ada kolom lokasi/region.** Status "warganet Blitar" bersandar pada konteks
   video (akun dan judul ber-Blitar), bukan domisili komentator. Jangan menulis
   "responden berdomisili di Blitar".
2. **Pengambilan dibatasi 200 komentar top-level per unggahan** (ditambah balasan),
   sehingga data ini sampel teratas per unggahan, bukan seluruh komentar. Ini juga
   alasan unggahan terbesar hanya 581 baris.
3. **Tidak ada video Mar–Agu 2026**, sehingga jumlah komentar bulanan tidak sebanding.
4. **Tiga video menyumbang ±52% data.** Distribusi tidak merata antar video.
5. **Kosakata Blitaran murni sangat sedikit.** Kata seperti `lare`, `mambu`, `elek`,
   `durung` nyaris tidak muncul (0–5 kali). Yang dominan bahasa gaul umum
   (`gak`/`ga`/`gk` 334 kali, `yg`, `aja`) dan sapaan ke akun (`min` 209, `kak` 226,
   `sppg` 517). Sebut **"gaul umum & Jawa Timuran"**, bukan "dialek Blitar".
6. **65% komentar tanpa like**, sehingga `diggCount` tidak layak dijadikan bobot
   model. Ia dipakai hanya untuk daftar "komentar paling disukai".
7. **47 baris (1,7%) ber-`clean_text` kosong.** Komentarnya hanya berisi emoji
   (mis. `🤣🤣🤣`), mention (`@admingerindraofficial`), atau huruf Unicode bergaya
   tebal/miring (`𝗦𝗽𝗽𝗴 𝘀𝘂𝗱𝗮𝗵 𝘁𝗮𝘂…`) yang ikut terhapus saat pembersihan karakter
   non-ASCII. Semuanya berlabel Netral (skor 0) dan tetap masuk split sebagai
   dokumen kosong. Sebutkan di keterbatasan; jangan dihitung sebagai "model gagal".

---

## 4. Alur pengolahan (pipeline)

### 4.1 Diagram alur

```
data/raw.json
   │  load_raw()            deteksi kolom, parse tanggal, normalisasi nama kolom
   ▼
filter_dataset()             rentang 1 Jan – 18 Nov 2026, buang kosong/pendek/duplikat
   ▼
preprocess_dataframe()       normalisasi → norm_text (untuk label)
   │                         stopword + stemming → clean_text (untuk TF-IDF)
   ▼
build_dataset()              skor leksikon hybrid → label Positif/Negatif/Netral
   ▼
split_dataset()              80% latih / 20% uji, stratify
   ▼
export_datasets()            dataset_processed_2026.csv, dataset_train.csv, dataset_test.csv
   ▼
train_and_evaluate()         TF-IDF → Naive Bayes & SVM → metrik + confusion matrix
   ▼
app.py                       dashboard 4 tab
```

### 4.2 Struktur berkas & kamus kolom

```text
Project/
├── app.py                     dashboard Streamlit (4 tab)
├── text_mining_pipeline.py    ingest → label → split → model
├── slang_dict.py              kamus slang, negasi, noise, emoji
├── extract_slang.py           Fase 1: pindai kosakata OOV
├── apify/                     bukti crawl: 7 tangkapan layar, source-video.txt,
│                              raw_apify_export.csv (ekspor CSV asli)
├── data/
│   ├── raw.json                       INPUT mentah 3.094 baris (9 kolom)
│   ├── dataset_processed_2026.csv     OUTPUT 2.779 baris berlabel
│   ├── dataset_train.csv              OUTPUT 2.223 baris
│   ├── dataset_test.csv               OUTPUT 556 baris
│   ├── validasi_manual.csv            OUTPUT 40 sampel
│   ├── slang_candidates.csv           OUTPUT 977 kandidat kosakata
│   ├── domain_lexicon.tsv             konfigurasi bobot kata MBG
│   └── positive.tsv, negative.tsv     kamus InSet
├── tests/                     22 tes
└── REFERENSI_LAPORAN_PPT.md   dokumen ini
```

Hanya `data/raw.json` yang berasal dari Apify; sisanya dihasilkan pipeline.

**Kolom `dataset_processed_2026.csv` / `dataset_train.csv` / `dataset_test.csv`**
(ketiganya berkolom identik):

| Kolom | Arti |
|---|---|
| `tanggal` | tanggal komentar `YYYY-MM-DD` |
| `komentar_asli` | teks mentah — **selalu kutip dari kolom ini** |
| `norm_text` | hasil cleaning + normalisasi slang + token emoji; belum buang stopword, belum stem |
| `clean_text` | `norm_text` → buang stopword → stemming; dipakai TF-IDF & WordCloud |
| `score` | skor polaritas hybrid |
| `label_sentimen` | Positif / Negatif / Netral |
| `split` | `latih` atau `uji` |
| `likes` | dari `diggCount` |
| `user` | `uniqueId` komentator |
| `video_url` | unggahan sumber |

**Kolom `validasi_manual.csv`:** `id`, `komentar_asli`, `clean_text`,
`label_hybrid` (aturan), `label_ai` (pembanding buatan AI), `tinjauan_manual`
(**masih kosong — harus diisi manusia**), `catatan`.

**Kolom `slang_candidates.csv`:** `kata`, `jumlah` (frekuensi di data mentah).

### 4.3 Penyaringan dan pembersihan data

| Tahap | Aturan | Hasil pada data ini |
|---|---|---|
| Baris di berkas mentah | — | 3.094 |
| Teks kosong | `text` kosong dibuang | −49 → 3.045 |
| Di luar jendela | di luar 1 Jan–18 Nov 2026 (data 2025) | −217 → 2.828 |
| Tanggal gagal | `pd.to_datetime(errors="coerce")` | 0 |
| Terlalu pendek | < 3 karakter | −35 → 2.793 |
| Duplikat | per `(user, text)` | −14 |
| **Tersisa** | | **2.779** |

Deduplikasi memakai pasangan pengguna + teks, **bukan** teks saja. Alasannya: ada 102
teks kembar di data mentah, tetapi hanya 21 yang benar-benar dari pengguna yang sama.
Contoh nyata: `"amin"` muncul 9 kali dari **9 pengguna berbeda** — itu 9 data sah, bukan
satu data yang digandakan. Sementara 14 baris yang terbuang di jendela 2026 adalah
komentar yang benar-benar sama dari akun yang sama (mis. terkirim dua kali).

### 4.4 Normalisasi teks (urutan tetap)

1. Lowercase.
2. Emoji unicode dan placeholder TikTok (`[Terkekeh]`, `[Sedih]`) diubah menjadi token
   `emojipos` / `emojineg` — **tidak dibuang**, karena emoji adalah sinyal sentimen kuat.
3. Hapus URL, `@mention`, `#hashtag`, angka, tanda baca, karakter non-ASCII.
4. Reduksi huruf berulang: `mantappp` → `mantap`.
5. Normalisasi kamus: `ora` → `tidak`, `apik` → `bagus`, `mbg` → `makan bergizi gratis`.
6. Stopword removal Sastrawi, **kecuali** kata negasi (`tidak`, `bukan`, `jangan`,
   `belum`, `kurang`, `tanpa`).
7. Stemming Sastrawi (hanya untuk `clean_text`).

### 4.5 Mengapa `norm_text` dan `clean_text` dipisah

| Kolom | Dipakai untuk | Alasan |
|---|---|---|
| `norm_text` | pelabelan leksikon | InSet berisi bentuk permukaan; stemming merusak pencocokan (`keracunan` → `racun`, tidak ada di kamus) |
| `clean_text` | TF-IDF, WordCloud | fitur mesin lebih baik pada kata dasar, dimensi lebih kecil |

Ini keputusan metodologis yang layak disebut di laporan.

---

## 5. Algoritma

### 5.1 Pelabelan hybrid enam lapis

InSet murni (pencocokan token biasa) **gagal** pada data ini. Hasil ujinya: 57% negatif,
tetapi itu artefak:

- `yang` berbobot **−5** (hanya ada di `negative.tsv`), `bagus` **+2 tetapi juga −4**,
  `enak` **+5 tetapi juga −4**.
- 1.142 kata muncul di `positive.tsv` **dan** `negative.tsv` dengan bobot berlawanan.
- `keracunan` — kata terpenting di dataset ini (58 komentar) — **tidak ada** di InSet.
- Contoh salah: `"enakk bangettt ayaammm nyaa sukaaa"` → skor −5 (negatif).

Karena itu dipakai enam lapis:

1. **InSet** — bobot kata yang muncul di dua berkas **dijumlahkan**, bukan ditimpa.
2. **Daftar kata-noise** — ±60 kata fungsi/frekuensi tinggi yang salah label dinolkan.
3. **Kamus domain MBG** (`data/domain_lexicon.tsv`) — `keracunan` −5, `telat` −5,
   `basi` −5, `bau`/`anyir`/`diare` −4, `seuprit`/`seculit` −2, `enak`/`mantap` +4.
   Kamus ini menimpa InSet.
4. **Flip negasi** — kata negasi membalik bobot kata positif dalam jendela 3 token:
   `tidak enak` → negatif.
5. **Emoji** — `emojipos` +1, `emojineg` −1.
6. **Ambang netral** — `|skor| ≤ 1` → Netral.

**Distribusi hasil:** Netral 1.408 (50,7%), Negatif 876 (31,5%), Positif 495 (17,8%).

### 5.2 TF-IDF

`TfidfVectorizer(max_features=2500, ngram_range=(1,2), sublinear_tf=True)`, **di-fit
hanya pada data latih** untuk mencegah kebocoran informasi dari data uji.

- TF: frekuensi kata dalam dokumen.
- IDF: `log(N / df)` — kata yang muncul di banyak dokumen diberi bobot kecil.
- `sublinear_tf`: `1 + log(tf)` meredam kata yang diulang-ulang.
- Bigram (1,2) menangkap frasa seperti `porsi kecil`, `tidak enak`.

### 5.3 Naive Bayes (Multinomial)

`P(kelas|dokumen) ∝ P(kelas) × Π P(kata|kelas)`, dengan `alpha = 0,1` (Laplace
smoothing) agar kata yang tak pernah muncul di kelas tertentu tidak membuat
probabilitas nol.

### 5.4 Support Vector Machine (LinearSVC)

Mencari hyperplane pemisah dengan margin maksimum:

```
minimalkan ½‖w‖²  dengan kendala  yᵢ(w·xᵢ + b) ≥ 1
```

`class_weight="balanced"` karena kelas tidak seimbang. Keunggulan pada teks: efektif
di ruang berdimensi tinggi dan tahan terhadap fitur yang tidak relevan.

### 5.5 Split dan stratifikasi

80% latih / 20% uji, `random_state=42`, `stratify=y`. Data latih **memang lebih banyak**
karena model belajar dari data latih, sedangkan data uji hanya dipakai sekali untuk
mengukur. Bila ada kelas berisi < 2 baris, kelas itu dibuang lebih dulu dan `stratify`
dilewati otomatis — keduanya dicatat di antarmuka.

### 5.6 Uji proporsi dua sampel

```
z = (p₁ − p₂) / √(p(1−p)(1/n₁ + 1/n₂)),   p = (x₁+x₂)/(n₁+n₂)
```

dipakai membandingkan proporsi sentimen paruh awal (Jan–Jun) dan paruh akhir
(Sep–Nov 2026).

---

## 6. Hasil

### 6.1 Distribusi sentimen keseluruhan (n = 2.779)

| Sentimen | Jumlah | Persentase |
|---|---|---|
| Netral | 1.408 | 50,7% |
| Negatif | 876 | 31,5% |
| Positif | 495 | 17,8% |

### 6.2 Sebaran per bulan (dengan jumlah, bukan hanya persentase)

| Bulan | n |
|---|---|
| Jan 2026 | 75 |
| Feb 2026 | 259 |
| Mar 2026 | 48 |
| Apr 2026 | 58 |
| Mei 2026 | 41 |
| Jun 2026 | 14 |
| Jul 2026 | 7 |
| Agu 2026 | 3 |
| Sep 2026 | 578 |
| Okt 2026 | 1.139 |
| Nov 2026 | 557 (parsial) |

**Cara menyajikan:** batang = jumlah komentar, garis = persentase sentimen, dan tiap
bulan diberi label n. Aplikasi sudah menyertakan anotasi "Mar–Agu tanpa video baru ·
Nov 2026 parsial". Jangan menampilkan hanya garis persentase — pembaca akan mengira
ada tren penurunan yang nyata, padahal penyebabnya jadwal publikasi.

**Kalau ingin menambah sendiri di PPT:** beri garis putus-putus pada Jun–Agu
(n = 14, 7, 3) untuk menandai bulan ber-volume sangat kecil. Itu batasan tampilan,
bukan perilaku bawaan aplikasi.

### 6.3 Perbandingan periode

| Periode | n | Positif | Negatif | Netral |
|---|---|---|---|---|
| Awal (Jan–Jun) | 495 | 21,2% | 23,4% | 55,4% |
| Akhir (Sep–Nov) | 2.274 | 17,1% | 33,4% | 49,5% |

- Negatif naik 10 poin: **p = 0,000015** (sangat signifikan).
- Positif turun 4 poin: **p = 0,031** (signifikan pada α = 0,05).
- Jul–Agu (10 komentar) dikeluarkan dari uji dan wajib disebut.

**Kalimat hasil:** *"Proporsi sentimen negatif meningkat signifikan dari 23,4% pada
paruh awal menjadi 33,4% pada paruh akhir 2026 (p < 0,001), sementara sentimen positif
menurun dari 21,2% menjadi 17,1% (p = 0,031). Namun periode akhir didominasi tiga video
berbeda topik, sehingga perbedaan ini tidak dapat murni diatribusikan pada perubahan
opini publik."*

### 6.4 Evaluasi model (data uji n = 556)

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | F1 (weighted) |
|---|---|---|---|---|---|
| Naive Bayes | 0,656 | 0,666 | 0,599 | 0,616 | 0,650 |
| **SVM (LinearSVC)** | **0,723** | **0,720** | **0,698** | **0,708** | **0,722** |

F1 per kelas pada data uji:

| Model | Negatif | Netral | Positif |
|---|---|---|---|
| Naive Bayes | 0,603 | 0,725 | 0,519 |
| SVM | 0,663 | 0,771 | 0,689 |

**Kesimpulan:** SVM unggul ±6,7 poin akurasi dan ±9,2 poin F1-macro. Penentu model
terbaik adalah **F1-macro**, bukan accuracy, karena kelas tidak seimbang — model yang
selalu menebak "Netral" akan tampak akurat tetapi tidak berguna.

**Pola yang layak dibahas:** kedua model paling lemah pada kelas **Positif** (F1 0,519
NB, 0,689 SVM). Confusion matrix-nya menunjukkan penyebabnya: 37 komentar Positif
ditebak Netral oleh NB dan 23 oleh SVM. Ini konsisten dengan temuan di 6.5 — batas
antara "pujian" dan "komentar biasa" memang paling kabur.

Angka lengkap per kelas (precision, recall, support) otomatis muncul di Tab 2 aplikasi;
salin dari sana bila dosen meminta.

### 6.5 Validasi label — tiga angka, tiga makna

Sampel 40 komentar sudah ditinjau manusia. Hasilnya:

| Perbandingan | Kesesuaian | Arti |
|---|---|---|
| Hybrid vs tinjauan manusia | **60%** (24/40) | seberapa benar aturan leksikon yang menjadi sumber label dataset |
| Label AI vs tinjauan manusia | **90%** (36/40) | seberapa dekat pembanding AI dengan penilaian manusia |
| Hybrid vs label AI | **55%** (22/40) | konsistensi aturan terhadap pembanding (bukan akurasi) |

**Cara membacanya — ini bagian penting untuk pembahasan:** pembanding AI yang
disusun terpisah ternyata **90% sepakat dengan manusia**, sedangkan aturan hybrid
hanya **60%**. Jadi kelemahan utama ada di **aturan leksikon**, bukan pada data
maupun pada pembandingnya. Ini argumen yang kuat dan jujur sekaligus.

Rincian hybrid vs penilaian manusia:

| | Manusia: Negatif | Netral | Positif |
|---|---|---|---|
| Hybrid: Negatif | 7 | 5 | 0 |
| Hybrid: Netral | 7 | 13 | 2 |
| Hybrid: Positif | 1 | 1 | 4 |

Pola kesalahannya: **7 komentar yang manusia nilai negatif justru dilabeli Netral**
oleh aturan hybrid, dan **5 komentar negatif dinilai Netral oleh manusia tetapi
dilabeli Negatif** oleh aturan. Artinya aturan hybrid kesulitan menempatkan kritik
yang disampaikan halus atau bernada sindiran.

Rincian label AI vs penilaian manusia:

| | Manusia: Negatif | Netral | Positif |
|---|---|---|---|
| AI: Negatif | 15 | 1 | 0 |
| AI: Netral | 0 | 18 | 3 |
| AI: Positif | 0 | 0 | 3 |

Hanya 4 dari 40 baris berbeda, dan semuanya bergeser satu tingkat (Netral ↔ Negatif,
Netral ↔ Positif) — tidak pernah terbalik kutub.

> **Yang boleh dan tidak boleh ditulis.** Boleh: *"label hybrid divalidasi manual
> pada 40 sampel dengan tingkat kesesuaian 60%"*. Jangan: *"akurasi pelabelan 90%"* —
> angka 90% itu kesesuaian **AI vs manusia**, bukan performa aturan yang dipakai.

---

## 7. Cara menggunakan aplikasi

### 7.1 Instalasi

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 7.2 Menjalankan

```bash
python extract_slang.py          # opsional: pindai kosakata OOV
python text_mining_pipeline.py   # wajib: hasilkan 3 CSV
streamlit run app.py
```

Opsi pipeline: `--biner` (buang kelas Netral), `--start` / `--end` (rentang tanggal).

### 7.3 Sidebar

| Kontrol | Fungsi |
|---|---|
| Unggah berkas | memuat `.csv`/`.json` mentah baru, lalu jalankan ulang pipeline |
| Kolom komentar / tanggal | dipakai bila nama kolom berbeda dari standar Apify |
| Rentang tanggal | default 1 Jan – 18 Nov 2026 |
| Sertakan kelas Netral | menyaring tampilan saja; model tidak dilatih ulang |
| Filter keyword lokal | mis. `telur`, `keracunan`, `anggaran` |
| Maks komentar per video | membatasi tampilan agar 3 video dominan tidak menenggelamkan sisanya |

### 7.4 Isi tab

| Tab | Isi | Untuk bab laporan |
|---|---|---|
| 1. Ringkasan & Waktu | KPI, grafik bulanan + volume, donut, uji periode | Hasil & pembahasan |
| 2. Evaluasi Model | tabel metrik, 2 confusion matrix, kesimpulan model terbaik, panel validasi (55% / 60% / 90%) | Hasil pengujian model |
| 3. Kata & Topik | WordCloud Positif/Negatif, **contoh komentar per kelas** (asli → normalisasi → bersih), 10 frasa teratas, tabel transformasi dialek, komentar terpopuler | Analisis konten |
| 4. Eksplorasi & Unduh | tabel interaktif, pencarian, unduh 4 CSV (opsi pemisah `;`) | Lampiran |

---

## 8. Kesimpulan yang bisa dipakai

1. Dari 3.094 komentar mentah, 2.779 lolos penyaringan. Sentimen didominasi **Netral
   (50,7%)**, diikuti **Negatif (31,5%)**, dan **Positif (17,8%)**. Dominasi Netral wajar
   karena banyak komentar berupa pertanyaan, permintaan menu, atau penandaan akun.
2. **SVM (LinearSVC) lebih baik daripada Naive Bayes** dengan akurasi 0,723 vs 0,656 dan
   F1-macro 0,708 vs 0,616.
3. Proporsi **sentimen negatif meningkat signifikan** pada paruh akhir 2026 (p < 0,001),
   sedangkan positif menurun (p = 0,031).
4. Pelabelan berbasis leksikon **belum seakurat penilaian manusia**: pada 40 sampel
   yang ditinjau, aturan hybrid hanya sepakat **60%**, sedangkan pembanding AI sepakat
   **90%**. Kelemahan ada di aturan, bukan di data.
5. Keterbatasan utama bukan pada model, melainkan pada **cakupan data**: tidak ada
   region komentator, pembatasan 200 komentar top-level per unggahan, tidak ada
   unggahan Mar–Agu, dan tiga unggahan mendominasi 52% data.

---

## 9. Saran pengembangan

1. Tambah video yang terbit Mar–Agu dan akun non-SPPG agar sebaran bulanan lebih rata.
2. Tambah kolom region komentator (bila scraper mendukung) untuk menguatkan klaim lokasi.
3. Anotasi manual pada ±300 komentar sebagai data latih *gold standard*.
4. Bandingkan dengan model berbasis transformer (IndoBERT) sebagai pembanding modern.
5. Tambah analisis topik (LDA) untuk memetakan isu: porsi, keracunan, anggaran, menu.

---

## 10. Kerangka PPT (12–13 slide)

| Slide | Isi |
|---|---|
| 1 | Judul, nama, program studi |
| 2 | Latar belakang: MBG dan reaksi publik di media sosial |
| 3 | Rumusan masalah & tujuan |
| 4 | Sumber data: aktor `clockworks/tiktok-comments-scraper`, 15 unggahan, 3.094 baris, batas 200 komentar top-level/unggahan |
| 5 | Batasan data: tanpa region, celah Mar–Agu, 3 video = 52%, kuota $5 habis |
| 6 | Alur pipeline (diagram dari bagian 4) |
| 7 | Mengapa InSet murni gagal → pelabelan hybrid 6 lapis |
| 8 | TF-IDF, Naive Bayes, SVM (rumus ringkas) |
| 9 | Hasil: distribusi sentimen + grafik bulanan dengan volume n |
| 10 | Hasil: tabel metrik + confusion matrix, SVM unggul |
| 11 | Validasi label: hybrid vs manusia 60%, AI vs manusia 90% (40 sampel) |
| 12 | Uji periode: negatif 23,4% → 33,4% (p < 0,001) |
| 13 | Kesimpulan, keterbatasan, saran |

Kalau harus tetap 12 slide, gabungkan slide 11 ke slide 10 sebagai catatan kaki.

Sisipkan **satu tangkapan layar dashboard** di slide 9 atau 10 — itu bukti aplikasi
berjalan. Satu tangkapan layar Apify (`apify/Screenshot …145850.png`, dialog ekspor
3.094 baris) juga layak masuk slide 4 sebagai bukti pengambilan data.

---

## 11. Pertanyaan penguji yang mungkin muncul

**"Kenapa data latih lebih banyak dari data uji?"**
Model belajar dari data latih; data uji hanya untuk mengukur sekali di akhir. Rasio
80/20 adalah konvensi umum dan permintaan spesifikasi tugas. Memberi porsi besar ke data
uji berarti membuang data yang seharusnya dipakai belajar.

**"Kenapa bulan Juni–Agustus hampir kosong?"**
Bukan karena opini berhenti, tetapi karena tidak ada unggahan baru yang terbit pada
periode itu (terverifikasi dari tanggal publikasi 15 unggahan). Komentar menumpuk
di sekitar unggahan baru: Jun 14, Jul 7, Agu 3 baris.

**"Kenapa pakai hybrid, bukan InSet saja?"**
InSet murni memberi 57% negatif yang artefak: `yang` berbobot −5, `bagus` punya
bobot +2 dan −4 sekaligus, dan `keracunan` tidak ada di kamus. Contoh konkret:
pujian "enak banget suka" terlabel negatif.

**"Akurasi 72% itu bagus atau jelek?"**
Cukup untuk tugas ini, tetapi perlu dibaca bersama F1-macro (0,708) karena kelas tidak
seimbang. Yang lebih penting: label dasarnya sendiri hanya 60% sepakat dengan
penilaian manusia, sehingga batas atas performa model terbatas oleh kualitas label.

**"Berarti label dataset-nya salah dong?"**
Sebagian, dan itu sudah diukur: 16 dari 40 sampel dinilai berbeda oleh manusia.
Kesalahannya berpola — kritik halus/sindiran cenderung dilabeli Netral. Inilah alasan
kesimpulan penelitian menyebut perlunya anotasi manusia sebagai pekerjaan lanjutan.

**"Apakah data ini mewakili warga Blitar?"**
Tidak sepenuhnya. Tidak ada informasi domisili komentator; lokasi disimpulkan dari
konteks video. Ini ditulis sebagai keterbatasan, bukan diklaim sebagai representasi
populasi.

**"Kenapa tidak pakai SMOTE?"**
SMOTE membuat data sintetis dari teks, yang tidak bermakna secara linguistik. Ketidak-
seimbangan ditangani lewat `class_weight="balanced"` dan pelaporan F1-macro.

**"Kenapa tidak menambah video atau komentar lagi?"**
Kuota gratis Apify ($5) sudah habis — terbukti pada tangkapan layar
`apify/Screenshot …145458.png`. Pengambilan ulang butuh akun/kuota baru.

**"Kenapa jumlah komentar tiap unggahan berbeda jauh?"**
Karena aktor diatur mengambil maksimum **200 komentar top-level per unggahan**
ditambah balasannya, bukan seluruh komentar. Lihat
`apify/Screenshot …145521.png`.

---

## 12. Pemetaan artefak ke bab laporan

| Bab | Berkas pendukung |
|---|---|
| Metodologi — pengumpulan data | `apify/` (7 tangkapan layar + `source-video.txt`), `data/raw.json` |
| Metodologi — struktur & kamus data | `README.md` bagian "Struktur folder" & "Kamus data", bagian 4.2 dokumen ini |
| Metodologi — preprocessing | `slang_dict.py`, `data/slang_candidates.csv`, kolom `norm_text` |
| Metodologi — pelabelan | `data/domain_lexicon.tsv`, `data/validasi_manual.csv` |
| Hasil — dataset | `data/dataset_processed_2026.csv`, `dataset_train.csv`, `dataset_test.csv` |
| Hasil — model | Tab 2 `app.py`, `tests/test_pipeline.py` |
| Lampiran | keempat CSV (Tab 4), tangkapan layar dashboard, tangkapan layar Apify |
