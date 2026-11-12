# Entri bounty make.fun — "Predict the top 5 new Solana coins by market cap, 24 hours out"

Program `main.py` (Python 3.13, **stdlib saja, tanpa network, deterministik**)
membaca history mark 10-menit semua koin pump.fun sejak pembukaan dan menulis
`forecast.json` yang dinilai log score pada 7 pembacaan per jam (20:00 UTC–
02:00 UTC). Hadiah butuh skor **di atas** forecast house (baseline starter kit).

## Cara menjalankan

```bash
# di dalam container grader (/input dan /output sudah ada):
python3 main.py

# di luar container (kit practice):
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Tidak ada argumen, tidak ada dependensi, tidak ada jam dinding: horizon
diambil dari `task.json` (`reads`) dan mark terakhir, jadi dua run pada input
yang sama menghasilkan byte yang identik (seed Monte Carlo tetap
`20261112`).

## Format input/output (disesuaikan dengan starter kit bounty)

Kit diunduh dari `https://make.fun/c/pumpfun-top5-forecast-01/kit` dan
formatnya diverifikasi terhadap `starter/main.ts` (baseline house) serta
`grader/forecast/ranked.ts` (scorer kanonik, `P_FLOOR = 1e-4`):

- **Input** `/input/task.json`: `places` (5), `output` (`forecast.json`),
  `size.edges` (81 edge log10 cap dalam SOL, 2.0–6.0 step 0.05 → 82 bin),
  `reads` (7 waktu jawab). `/input/history.ndjson`: mark `{at, ok, rows}`,
  hanya mark `ok` dengan `rows` non-kosong yang dipakai; baris `{rank, mint,
  market_cap_lamports, venue, slot, name, symbol, created_at, complete,
  mayhem, quote_reserves_lamports}`.
- **Output** `/output/forecast.json` (maks 1.048.576 byte), tepat tiga key:
  - `probabilities`: baris `{mint, place, p}` (place 1–5, p difloor ke
    6 desimal, tiap place jumlahnya ≤ 1).
  - `top5`: 5 mint berbeda, place 1 dulu.
  - `sizes`: baris `{place, bin, p}` (bin 0–81, `bin =` jumlah edge ≤ nilai;
    nilai = log10 market cap dalam SOL = log10(lamports) − 9; tiap place
    jumlahnya ≤ 1).
- **Aturan scorer yang dipatuhi**: tidak ada baris p = 0 (koin bernama
  dengan p 0 akan kena floor bila menang dan keluar dari jatah unnamed);
  tidak ada baris yang menamai mint yang tidak ada di history (scorer
  membuangnya sebelum menilai); massa yang tidak dinamai dibagi rata oleh
  scorer ke koin chart jawaban yang tidak dinamai — termasuk koin yang
  lahir setelah cut — jadi massa itu dijaga tetap jujur, bukan disebar
  tipis ke koin bernama.

## Cara program menghitung probabilitas

Koin diidentifikasi **hanya lewat mint address, bukan nama/symbol**: dua
koin bisa bernama sama (riset menemukan dua GABOON) dan hanya mint yang
membedakannya.

1. **Trajektori** tiap mint dibangun dari semua mark (log10 market cap
   dalam lamports). Koin yang sudah tidak muncul di mark terakhir tetap
   dihitung dengan drift global selama jedanya.
2. **Forecast per koin per read**: `E[logM(t)] = last + drift0·TAU·(1−e^(−t/TAU))
   + global·(t − TAU·(1−e^(−t/TAU)))`, `TAU = 2,5 jam`. `drift0` = slope
   log koin sendiri (0,3× slope 1 jam + 0,7× slope 3 jam) yang disusutkan
   65% ke arah drift global. Drift global = median slope 3 jam dari **10
   koin teratas** di cut, dijepit ke [−0,015, +0,005]/jam — median semua koin
   adalah koin ekor yang sekarat dan akan menyeret semua leader turun
   ~0,1–0,2 log10 selama window. Ketidakpastian `sigma(t) =
   sqrt(sigma_h²·t + sigma0²)`; `sigma_h` dari **median** return log
   10-menit absolut (varians akan membuat ramp peluncuran terlihat seperti
   volatilitas berjalan — itu sempat mendorong sigma ke cap untuk semua
   koin yang bergerak cepat), dijepit [0,12, 0,50] log10 per √jam.
3. **Risiko collapse**: tiap koin punya hazard per jam (muda / likuiditas
   tipis vs market cap / belum lulus menaikkannya; lulus menurunkannya).
   Koin yang masih di curve bisa di-rug (−1,7 log10); koin yang sudah lulus
   duduk di pool nyata, jadi lantainya adalah drawdown dalam (−0,6 log10).
   Inilah yang membuat leader 12× lipat tidak mendapat P(1) ≈ 0,97 —
   drawdown 75% masih ia tahan, rug tidak.
4. **Monte Carlo** 20.000 path × 7 read (seed tetap) atas ≤ 60 kandidat
   (top 60 by expected cap di read pertama + 10 riser 1 jam tercepat):
   tiap path mengundi log market cap tiap kandidat di tiap read, memberi
   peringkat, dan mencatat pemenang tiap place beserta log market cap-nya.
   Frekuensi simulasi × (1 − q_other[place]) = probabilitas place.
5. **q_other** (massa untuk koin di luar kandidat + koin lahir setelah cut)
   diukur dari history sendiri: pada semua pasangan mark berjarak ~7 jam,
   fraksi slot top-5 di mark akhir yang dipegang koin di luar top-60 mark
   awal; dijepit [0,02, 0,30], fallback 0,08 bila history terlalu pendek.
6. **Sizes**: distribusi pemenang log10 cap (SOL) per place dari simulasi
   yang sama, dihistogramkan ke 82 bin. **top5**: expected log market cap
   tertinggi di read tengah.

## Hasil tes (12 Nov 2026)

Tes ada di `../tests/` (`gen_synth.py` = generator data sintetis,
`score.py` = validator format + scorer port dari `ranked.ts` + port
baseline starter untuk perbandingan).

**Data sintetis** (67 mark, 02:00–13:00 UTC, 39 koin; trajektori dikalibrasi
pada snapshot riset ~05:43 UTC — GABOON#1 $2,43 jt, DacityCoin ~$210 rb,
FLC, GABOON#2, FORKYU jatuh; plus pendatang baru MOONX yang lahir setelah
cut dan masuk top-5 — menguji penanganan unnamed mass):

| | order | sizes | TOTAL | agreement |
|---|---|---|---|---|
| **program ini** | −7,470 | −13,322 | **−20,792** | 130 |
| baseline house | −9,434 | −14,780 | −24,213 | 130 |
| **selisih** | +1,96 | +1,46 | **+3,42 nats** | sama |

- Format **VALID** (port `parseRanked`/`parseSizes`): 189 baris
  probabilities, 214 baris sizes, 24.089 byte; jumlah per place 0,7000
  (q_other 0,30 — churn terukur tinggi di data sintetis); jumlah sizes
  1,0000.
- **Deterministik**: dua run menghasilkan file byte-identik (`cmp` OK).
- **Runtime ~4,6–4,8 detik** (batas 300 detik).
- top5 program = 5 prediksi riset (GABOON#1, DacityCoin, FLC, GABOON#2,
  SCAMSON); P(1) GABOON#1 = 0,64.

**Input kit practice asli** (`/tmp/kit/pumpfun-top5-forecast-01-kit/input`,
6 mark, ~235 koin): jalan, format VALID (82 baris probabilities, 121 baris
sizes, 11.975 byte), deterministik (byte-identik), runtime ~6,7 detik.
top5 sama dengan baseline di #1/#2/#5, berbeda di #3/#4 (momentum vs
persistensi). Skor practice tidak bisa dihitung — outcome read
21:00–22:00 tidak ada di kit (skor baseline −40,408 di RULES.txt memakai
dataset sealed).

## Catatan / asumsi

- `task.json` diasumsikan memuat `reads`; bila tidak ada, dipakai 7 read
  per jam mulai 7 jam setelah cut.
- Bila history hanya berisi 1 jam mark (seperti practice kit), pasangan
  mark 7 jam untuk mengukur q_other tidak ada → fallback q_other = 0,08.
- Program tidak memakai nama/symbol, harga SOL, atau data eksternal apa
  pun: semua dari `/input/history.ndjson`.
- Bukan hardcode daftar statis: pada data baru, kandidat, drift,
  volatilitas, hazard, dan q_other semuanya dihitung ulang dari history.

## File

- `main.py` — program entri (satu file, stdlib saja).
- `README.md` — dokumen ini.
- `LICENSE` — lisensi MIT (disalin dari starter kit).
