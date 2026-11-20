# Analisis Sentimen MBG Blitar 2026

Pipeline text mining + dashboard Streamlit untuk komentar TikTok tentang program
Makan Bergizi Gratis (MBG) di Blitar sepanjang 2026.

## Instalasi

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Menjalankan

```bash
python extract_slang.py          # Fase 1: pindai kosakata OOV -> data/slang_candidates.csv
python text_mining_pipeline.py   # Fase 2: bersihkan, label, split, ekspor
streamlit run app.py
```

Opsi pipeline: `--biner` (buang kelas Netral), `--start` / `--end` (rentang tanggal).
Letakkan berkas mentah `.csv` / `.json` hasil Apify di folder `data/`.

## Struktur folder

```text
Project/
├── app.py                     Dashboard Streamlit (4 tab)
├── text_mining_pipeline.py    Seluruh logika: ingest → label → split → model
├── slang_dict.py              SLANG_DICT, NEGATION_WORDS, NOISE_WORDS, EMOJI_MAP
├── extract_slang.py           Fase 1: pindai kosakata OOV dari data mentah
│
├── apify/                     Bukti pengambilan data (untuk lampiran laporan)
│   ├── Screenshot *.png       Tangkapan layar console Apify
│   ├── raw_apify_export.csv   Ekspor CSV asli dari Apify (3.094 baris)
│   └── source-video.txt       15 URL unggahan sumber
│
├── data/
│   ├── raw.json                          INPUT mentah (3.094 baris, 9 kolom)
│   ├── dataset_processed_2026.csv        OUTPUT 2.779 baris berlabel
│   ├── dataset_train.csv                 OUTPUT 2.223 baris (80%)
│   ├── dataset_test.csv                  OUTPUT 556 baris (20%)
│   ├── validasi_manual.csv               OUTPUT 40 sampel pemeriksaan label
│   ├── slang_candidates.csv              OUTPUT 977 kandidat kosakata
│   ├── domain_lexicon.tsv                Konfigurasi: bobot kata domain MBG
│   └── positive.tsv / negative.tsv       Kamus InSet (diunduh otomatis)
│
├── REFERENSI_LAPORAN_PPT.md   Acuan penulisan laporan & slide
├── tests/                    23 tes (pipeline + smoke test aplikasi)
└── requirements.txt
```

Hanya **`data/raw.json`** yang perlu Anda sediakan. Sisanya dihasilkan otomatis
oleh `text_mining_pipeline.py`. Jangan mengedit CSV hasil olahan dengan tangan —
jalankan ulang pipeline bila ingin mengubah sesuatu. Satu pengecualian:
`data/validasi_manual.csv` memang untuk diisi tangan pada kolom `tinjauan_manual`.

## Kamus data

### Input mentah — `data/raw.json`

Berkas asli Apify, satu objek per komentar (3.094 baris). Kolomnya:

| Kolom | Arti |
|---|---|
| `text` | isi komentar; 49 baris kosong |
| `diggCount` | jumlah suka pada komentar |
| `replyCommentTotal` | jumlah balasan komentar (kosong pada 1.755 baris) |
| `createTimeISO` | waktu komentar (ISO 8601, UTC) |
| `uniqueId` | nama pengguna komentator |
| `videoWebUrl` | URL unggahan sumber |
| `uid`, `cid` | ID pengguna & ID komentar dari TikTok |
| `avatarThumbnail` | URL foto profil |

### Dataset olahan — `dataset_processed_2026.csv`, `dataset_train.csv`, `dataset_test.csv`

Ketiganya **berkolom identik** (10 kolom); bedanya hanya cakupan baris.

| Kolom | Arti |
|---|---|
| `tanggal` | tanggal komentar, `YYYY-MM-DD`, dari `createTimeISO` |
| `komentar_asli` | teks mentah apa adanya (untuk kutipan laporan) |
| `norm_text` | setelah cleaning + normalisasi slang + token emoji; **belum** dibuang stopword & belum di-stem |
| `clean_text` | `norm_text` → buang stopword → stemming Sastrawi; dipakai TF-IDF & WordCloud |
| `score` | skor polaritas hybrid (negatif = negatif, 0 = netral) |
| `label_sentimen` | `Positif` / `Negatif` / `Netral` |
| `split` | `latih` (80%) atau `uji` (20%) |
| `likes` | dari `diggCount` |
| `user` | `uniqueId` komentator |
| `video_url` | unggahan sumber |

Ringkasan baris: `processed` 2.779 = `train` 2.223 + `test` 556.

Contoh satu baris:

| Kolom | Nilai |
|---|---|
| `komentar_asli` | `min, mbok ya tulung yang ayamnya secuil itu ditambahin.. muridku pas dapet jan secuil seuprit, la teman sebangkunya dapat besar ... 😳😳😳` |
| `norm_text` | `admin coba ya tulung yang ayamnya secuil itu ditambahin muridku pas dapat jan secuil sedikit la teman sebangkunya dapat besar` |
| `clean_text` | `admin coba tulung ayam cuil ditambahin murid pas jan cuil sedikit la teman bangku besar` |
| `score` / `label_sentimen` | `5.0` / `Positif` |

Contoh ini memperlihatkan dua hal: `mbok` → `coba`, `seuprit` → `sedikit` (kamus
slang bekerja), tetapi `ditambahin` tidak menjadi `tambah` dan `ayamnya` menjadi
`ayam` — stemming Sastrawi tidak selalu tuntas. Untuk laporan, kutipan selalu
diambil dari `komentar_asli`, bukan dari `clean_text`.

### `data/validasi_manual.csv` (40 baris)

| Kolom | Arti |
|---|---|
| `id` | nomor sampel 1–40 |
| `komentar_asli`, `clean_text` | teks sampel |
| `label_hybrid` | label hasil aturan leksikon |
| `label_ai` | label pembanding **buatan AI** (bukan manusia) |
| `tinjauan_manual` | **kosong — tugas Anda**: isi `sesuai` / `tidak` |
| `catatan` | alasan bila `tidak` sesuai |

### `data/slang_candidates.csv` (977 baris)

| Kolom | Arti |
|---|---|
| `kata` | token yang belum ada di `slang_dict.py` |
| `jumlah` | frekuensi kemunculan di data mentah |

Bukan keluaran analisis, melainkan bahan untuk menambah `SLANG_DICT`.

## Bukti pengambilan data — folder `apify/`

Isi folder ini untuk lampiran bab metodologi, bukan untuk dibaca program:

| Berkas | Isi |
|---|---|
| `Screenshot 2026-11-20 145458.png` | Halaman *Input* aktor: 15 URL sumber + peringatan kuota $5 habis |
| `Screenshot 2026-11-20 145521.png` | Pengaturan batas crawl: maks 2.000 komentar/posting, 200 komentar top-level/posting, 100 balasan/komentar |
| `Screenshot 2026-11-20 145753.png` | Tabel *Output*: 9 kolom mentah, contoh unggahan SPPG Mojorejo |
| `Screenshot 2026-11-20 145802.png` | Tabel *Output*: contoh unggahan `storymasibin` |
| `Screenshot 2026-11-20 145850.png` | Dialog *Export dataset*: format CSV, 3.094 baris, run terakhir 5 m 30 s / $3,868 |
| `Screenshot 2026-11-20 145554.png`, `…145815.png` | Tangkapan layar pendukung proses crawl |
| `source-video.txt` | 15 URL unggahan beserta kata kunci pencariannya |

Dua hal yang bisa dikutip dari bukti ini: aktor yang dipakai adalah
**`clockworks/tiktok-comments-scraper`** (dijalankan lewat Apify Console), dan
**kuota gratis $5 sudah habis** sehingga pengambilan data tidak bisa diulang.

**Batas crawl 200 komentar top-level per posting** berarti data ini bukan
himpunan lengkap, melainkan sampel teratas per unggahan. Tulis sebagai
keterbatasan, bukan sebagai "seluruh komentar".

## Cara kerja pelabelan

Hybrid enam lapis: InSet (bobot kata ganda dijumlahkan), daftar kata-noise,
kamus domain MBG (`data/domain_lexicon.tsv`), flip negasi berjendela 3 token,
emoji sebagai token bernilai, dan ambang netral `|skor| <= 1`.

Pada data ini distribusinya sekitar **51% Netral, 31% Negatif, 18% Positif**.

## Validasi label — baca sebelum mengutip angka

`data/validasi_manual.csv` berisi 40 sampel. Ada **tiga angka berbeda** di Tab 2,
dan ketiganya bukan hal yang sama:

| Angka | Artinya | Nilai pada data ini |
|---|---|---|
| Hybrid vs label AI | konsistensi aturan leksikon terhadap pembanding buatan AI | **55%** |
| Hybrid vs tinjauan manusia | kesesuaian aturan dengan penilaian Anda pada 40 sampel | **60%** |
| Label AI vs tinjauan manusia | seberapa dekat pembanding AI dengan penilaian Anda | **90%** |

Cara membacanya: aturan hybrid yang menjadi **sumber label dataset** hanya sepakat
60% dengan penilaian manusia, sedangkan pembanding AI justru 90% sepakat. Artinya
kelemahan ada di **aturan leksikon**, bukan di data atau di pembandingnya. Ini
temuan yang layak dibahas, bukan sesuatu yang perlu disembunyikan.

Kolom `tinjauan_manual` **sudah terisi 40 baris** dengan `negatif` / `netral` /
`positif` (boleh juga `sesuai` / `tidak`). Bila Excel menyimpan berkas ini dengan
pemisah titik koma, aplikasi tetap membacanya dengan benar — pemisah dideteksi
otomatis. Yang perlu dijaga hanya: jangan mengubah kolom `komentar_asli`,
`label_hybrid`, dan `label_ai`.

## Batasan data yang harus ditulis di laporan

- **Tidak ada kolom lokasi/region**; "Blitar" bersandar pada konteks video,
  bukan domisili komentator.
- **Pengambilan dibatasi 200 komentar top-level per unggahan**, jadi ini sampel
  teratas, bukan seluruh komentar.
- **Tidak ada unggahan yang terbit Mar–Agu 2026**, sehingga bulan-bulan itu
  hampir kosong. Kurva bulanan mencerminkan jadwal publikasi, bukan opini.
- **Tiga unggahan menyumbang ±52% data** (581, 529, 461 komentar).
- **Kosakata Blitaran murni sangat sedikit**; normalisasi didominasi bahasa gaul
  umum dan Jawa Timuran.
- **47 baris (1,7%) punya `clean_text` kosong**, karena komentarnya hanya berisi
  emoji, mention, atau huruf Unicode bergaya tebal/miring yang terhapus saat
  cleaning. Semuanya berlabel Netral dan tetap ikut split.
- Analisis temporal bersifat deskriptif, ditambah uji awal (Jan–Jun) vs akhir
  (Sep–Nov).

## Merapikan berkas

Duplikat sudah dibersihkan: salinan JSON di root (identik dengan `data/raw.json`)
dan sisa `data/upload_*.csv` dihapus; ekspor CSV asli Apify dipindahkan ke
`apify/raw_apify_export.csv`. Sekarang **satu sumber data saja**: `data/raw.json`.

## Tes

```bash
python -m pytest -q
```

## Membuka CSV di Excel

Berkas keluaran memakai pemisah koma dan encoding UTF-8 dengan BOM. Bila Excel
berbahasa Indonesia menumpuk semua kolom menjadi satu, buka lewat
**Data → From Text/CSV → pilih Comma**. Di Tab 4 tersedia opsi mengunduh dengan
pemisah titik koma (`;`) untuk locale Indonesia.
