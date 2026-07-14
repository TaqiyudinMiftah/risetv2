# Research Execution Roadmap

## Scope Decision

Penelitian tidak akan mereproduksi penuh semua paper sebelum hipotesis utama
diuji. Angka paper lain sering memakai split, preprocessing, backbone, atau
training budget yang berbeda, sehingga tidak valid sebagai controlled
comparison. Reproduksi penuh juga memakai compute besar tanpa selalu mengisolasi
kontribusi bidirectional interaction dan post-interaction debiasing.

Eksperimen dibagi menjadi empat kelompok:

1. **Reproduksi penuh wajib:** upstream-community CAER-Net dan clean in-repo
   CAER-Net. CAER-Net + CCIM menjadi reproduksi penuh hanya jika kode,
   intervention, dan protokolnya dapat dicocokkan; jika tidak, namanya
   `CCIM-inspired`.
2. **Baseline komponen wajib:** face-only, context-only, concatenation, adaptive
   fusion, unidirectional cross-attention, bidirectional cross-attention,
   iterative cross-attention, raw/pre/post-interaction debiasing, dan CD-ICA-Net.
3. **Reproduksi penuh opsional:** GLAMOR-Net, CAHFW-Net, CLEF, DSCT, dan AGCD-Net.
   Pekerjaan ini dimulai hanya setelah ablation utama selesai dan official code
   serta protokolnya cukup lengkap.
4. **Reported-only literature:** hasil publikasi ditempatkan di tabel literatur,
   terpisah dari hasil reproduksi dan controlled comparison.

## Execution Order

1. **Protocol and baseline audit - complete.** Split
   `caer_s_content_disjoint_v1`, detector, overlap, class order, dan evaluator
   telah diaudit. Test set masih terkunci.
2. **Freeze CAER-Net - in progress.** Run eksplorasi seed 42 mencapai validation
   accuracy `0.7271` dan macro F1 `0.7299`. Protokol final dibekukan pada seed
   42, 43, 44 dengan budget 45 epoch, early stopping 12, dan seleksi
   `val_accuracy` untuk track upstream-community.
3. **Clean in-repo baseline.** Pindahkan business logic notebook ke modul,
   verifikasi prediksi, lalu jalankan tiga seed. Track ini tidak digabung dengan
   hasil upstream-community.
4. **Input and bias diagnostics.** Jalankan face-only, context-only, masked,
   neutralized, shuffled, dan swapped-context diagnostics tanpa menganggapnya
   sebagai bukti kausal.
5. **Interaction ablation.** Bandingkan concat/adaptive fusion, one-way,
   bidirectional, dan iterative attention untuk `N=1,2,3`.
6. **Debiasing ablation.** Bandingkan no debiasing, raw/pre/post-interaction,
   CCIM atau CCIM-inspired, dengan seluruh faktor training lain tetap.
7. **CD-ICA-Net and final evidence.** Bekukan kandidat, jalankan tiga seed,
   kemudian buka test sekali untuk accuracy, macro/per-class F1, calibration,
   robustness, statistik, latency, parameter, dan memory.
8. **Paper package.** Hasil akhir mencakup tabel utama dan ablation, confusion
   matrix, reliability diagram, attention maps, error analysis, registry, config,
   prediction, hash, dan checkpoint.

## Current Compute Gate

Tiga run final CAER-Net memerlukan compute panjang dan dijalankan berurutan
setelah dry-run berhasil. Tidak ada metrik test yang diakses pada tahap ini.
