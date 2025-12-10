# 🏨 Hotel Booking Cancellation Predictor

Ringkasan singkat proyek Streamlit untuk memprediksi pembatalan booking hotel dan memberikan rekomendasi tindakan.

Files utama:
- [(Final_Project_Hotel_Demand_JCDSAH_0208.ipynb)] - File utama untuk mengolah data
- [app.py](app.py) — Aplikasi Streamlit (GUI & logika prediksi)
- [helper.py](helper.py) — Skrip pembantu untuk mengekstrak nama fitur dari model
- [requirements.txt](requirements.txt) — Dependencies
- [model_features.csv](model_features.csv) — Daftar fitur yang dipakai model
- [feature_importance.csv](feature_importance.csv) — (opsional) ranking fitur
- `random_forest_best_model.pkl`, `scaler.pkl`, `label_encoders.pkl` — model & artefak (harus ada di root)

Prasyarat
1. Python 3.8+
2. Install dependency:
```bash
pip install -r requirements.txt
```

Menjalankan aplikasi (lokal)
```bash
streamlit run app.py
```

Ringkasan fungsi penting (lihat detail di file app):
- [`app.load_artifacts`](app.py) — memuat model, scaler, label encoders, dan daftar fitur.
- [`app.create_features`](app.py) — membuat fitur engineer dari input mentah.
- [`app.preprocess_input`](app.py) — preprocessing + encoding lalu menyusun DataFrame sesuai urutan fitur.
- [`app.get_risk_category`](app.py) — mengubah probabilitas menjadi kategori risiko.
- [`app.get_recommendations`](app.py) — menghasilkan rekomendasi tindakan berdasarkan probabilitas & konteks.
- [`app.main`](app.py) — UI Streamlit (Single Prediction, Batch Prediction, Model Info, About).

Penggunaan helper.py
- Jalankan untuk mengekstrak nama fitur dari model terlatih dan menyimpan ke `model_features.csv`:
```bash
python helper.py
```
(Lihat penjelasan lengkap di [helper.py](helper.py))

File konfigurasi & deployment
- Panduan deployment tersedia di [DEPLOYMENY_GUIDE.md](DEPLOYMENY_GUIDE.md). Rekomendasi: Streamlit Cloud / Render.
- Pastikan `.gitignore` mengecualikan env, dataset besar, dan file sensitif (lihat panduan).

Troubleshooting singkat
- Error file model tidak ditemukan → pastikan `random_forest_best_model.pkl`, `scaler.pkl`, `label_encoders.pkl` ada di root.
- Jika `model_features.csv` tidak ada → jalankan [`helper.py`](helper.py).
- Jika encountering unseen categorical values → label encoder fallback di-handle di [`app.preprocess_input`](app.py) (nilai di-set ke 0).

Lisensi & Disclaimer
- Model dibuat untuk tujuan edukasi & riset. Gunakan sebagai alat bantu keputusan, bukan keputusan final.

# app.py — Quick Reference

Lokasi: [app.py](app.py)

Ringkasan
- Aplikasi Streamlit interaktif untuk memprediksi probabilitas pembatalan booking hotel dan menampilkan rekomendasi serta visualisasi.

Entry point
- Fungsi utama: [`app.main`](app.py) — mem-build UI dan menjalankan alur aplikasi.

Alur utama
1. Load artefak: [`app.load_artifacts`](app.py)
   - Memuat: model (`random_forest_best_model.pkl`), `scaler.pkl`, `label_encoders.pkl`, dan `model_features.csv`.
2. Preprocessing: [`app.preprocess_input`](app.py)
   - Memanggil [`app.create_features`](app.py) → menambah fitur ter-engineer (lead_time flag, total_guests, commitment_score, risk_score, dll).
   - Meng-encode kategori dengan `label_encoders` dan menyusun DataFrame sesuai `feature_names`.
3. Prediksi & output:
   - model.predict / model.predict_proba → probabilitas pembatalan.
   - [`app.get_risk_category`](app.py) untuk label risiko.
   - [`app.get_recommendations`](app.py) untuk saran tindakan.
4. UI pages:
   - Single Prediction — form input manual
   - Batch Prediction — upload CSV & tombol "Predict All"
   - Model Info — metrik & feature importance
   - About — dokumentasi proyek

Input field mapping (Single Prediction)
- hotel, arrival_date, lead_time, adults, children, babies, stays_in_weekend_nights, stays_in_week_nights, meal, country, market_segment, distribution_channel, reserved_room_type, assigned_room_type, deposit_type, customer_type, adr, total_of_special_requests, previous_cancellations, previous_bookings_not_canceled, is_repeated_guest, booking_changes, required_car_parking_spaces

Tips debugging
- Jika error "Model not loaded" → cek file model/artefak ada di root.
- Untuk CSV batch, pastikan kolom nama sesuai atau lakukan penyesuaian sebelum upload.

# helper.py — Quick Reference

Lokasi: [helper.py](helper.py)

Fungsi
- Skrip sederhana untuk mengekstrak daftar nama fitur dari model terlatih dan menyimpannya ke `model_features.csv`.

Cara pakai
```bash
python helper.py
```

Apa yang dilakukan
1. Membuka file `random_forest_best_model.pkl`.
2. Jika model memiliki atribut `feature_names_in_`, skrip akan:
   - Mengekstrak nama fitur → menyimpan ke `model_features.csv`.
   - Menampilkan 10 fitur pertama di console.
3. Menangani error jika file model tidak ditemukan atau atribut tidak tersedia.

Catatan
- Nama file model yang dicari di skrip: `random_forest_best_model.pkl`. Jika model Anda punya nama berbeda, ubah nama file di `helper.py` atau rename model Anda.
- Jika model tidak memiliki `feature_names_in_`, app Streamlit masih akan berjalan (app menambahkan fallback untuk fitur yang hilang).
