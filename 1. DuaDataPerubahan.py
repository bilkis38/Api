import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from joblib import dump
import time

# 1. Baca Data Pertama
data1 = pd.read_csv('data__2.csv')

# 2. Baca Data Kedua
data2 = pd.read_csv('data__3.csv')

# 3. Gabungkan Data
data = pd.concat([data1, data2], ignore_index=True)

# 4. Proses Data
data['Time'] = pd.to_datetime(data['Time'], errors='coerce')

# 5. Hitung Perubahan
kolom_monitor = ['Acceleration X(g)', 'Acceleration Y(g)', 'Acceleration Z(g)',
                 'Angular velocity X(°/s)', 'Angular velocity Y(°/s)', 'Angular velocity Z(°/s)',
                 'Angle X(°)', 'Angle Y(°)', 'Angle Z(°)', 'Temperature(℃)']

for kolom in kolom_monitor:
    data[f'{kolom}_perubahan'] = data[kolom].diff()

# 6. Tambahkan Kolom 'Device name' dengan Nilai Default 'com4'
data['Device name'] = 'com4'

# 7. Analisis Perubahan
batas_ambang = 0.1
perubahan_signifikan = data[data['Acceleration X(g)_perubahan'].abs() > batas_ambang]

# 8. Buat DataFrame Baru hanya untuk Data Perubahan
hasil_perubahan = perubahan_signifikan[['Time', 'Device name', 'Acceleration X(g)_perubahan', 'Acceleration Y(g)_perubahan', 'Acceleration Z(g)_perubahan', 'Angular velocity X(°/s)_perubahan', 'Angular velocity Y(°/s)_perubahan', 'Angular velocity Z(°/s)_perubahan', 'Angle X(°)_perubahan', 'Angle Y(°)_perubahan', 'Angle Z(°)_perubahan']].copy()

# 9. Ganti nama kolom hasil perubahan tanpa kata "perubahan" di belakangnya
hasil_perubahan.columns = [col.replace('_perubahan', '') for col in hasil_perubahan.columns]

# 10. Pisahkan data asli dan hasil perubahan
data_asli = data.drop(perubahan_signifikan.index)

# 11. Pisahkan data hasil perubahan yang ingin ditampilkan di Excel
data_perubahan_yang_akan_disimpan = hasil_perubahan[['Time', 'Device name', 'Acceleration X(g)', 'Acceleration Y(g)', 'Acceleration Z(g)', 'Angular velocity X(°/s)', 'Angular velocity Y(°/s)', 'Angular velocity Z(°/s)', 'Angle X(°)', 'Angle Y(°)', 'Angle Z(°)']]

# 12. Konversi Data Perubahan yang ingin ditampilkan ke Excel
file_excel_hasil_perubahan = 'hasil_perubahan.xlsx'
data_perubahan_yang_akan_disimpan.to_excel(file_excel_hasil_perubahan, index=False)
print(f"Hasil perubahan yang ingin ditampilkan telah disimpan ke {file_excel_hasil_perubahan}")

# 13. Tampilkan data hasil perubahan di layar monitor
print("Data Hasil Perubahan yang ingin ditampilkan:")
print(data_perubahan_yang_akan_disimpan)

# 14. Persiapkan Data untuk Model ANN
# Tentukan fitur dan target
fitur = data_perubahan_yang_akan_disimpan.drop(['Time', 'Device name'], axis=1)
target = (hasil_perubahan['Acceleration X(g)'] > 0.1).astype(int)  # Contoh target, sesuaikan dengan kondisi yang ingin Anda prediksi

# Bagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(fitur, target, test_size=0.2, random_state=42)

# 15. Inisialisasi dan Latih Model ANN
model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)

# Melatih model
print("Proses pelatihan model dimulai...")
for i in range(0, 500, 5):  # 500 iterasi dengan jeda setiap 10 detik
    time.sleep(5)  # Jeda 10 detik
    model.partial_fit(X_train, y_train, classes=np.unique(y_train))
    print(f"Proses pelatihan: {i} iterasi selesai")

print("Pelatihan model selesai.")

# Simpan model ke file .joblib
dump(model, 'model_ann.joblib')

# 16. Prediksi Data Uji
predictions = model.predict(X_test)

# 17. Evaluasi Model
accuracy = accuracy_score(y_test, predictions)
print(f'Akurasi Model: {accuracy}')

# 18. Prediksi Data Asli yang Belum Diproses
fitur_data_asli = data_asli[['Time', 'Acceleration X(g)', 'Acceleration Y(g)', 'Acceleration Z(g)',
                             'Angular velocity X(°/s)', 'Angular velocity Y(°/s)', 'Angular velocity Z(°/s)',
                             'Angle X(°)', 'Angle Y(°)', 'Angle Z(°)']]

# Prediksi perubahan menggunakan model
predictions_data_asli = model.predict(fitur_data_asli.drop('Time', axis=1))

# 19. Gabungkan prediksi dengan data asli
data_asli['Predicted Change'] = predictions_data_asli

# 20. Tambahkan kolom 'Time' ke hasil akhir
data_asli['Time'] = data_asli['Time'].dt.strftime('%Y-%m-%d %H:%M:%S')

# Tambahkan kolom 'Time' default jika kolom tersebut tidak ada
if 'Time' not in data_asli.columns:
    data_asli['Time'] = pd.Timestamp('now').strftime('%Y-%m-%d %H:%M:%S')

# 21. Simpan Hasil ke Excel
file_excel_hasil_akhir = 'hasil_akhir.xlsx'
data_asli.to_excel(file_excel_hasil_akhir, index=False)
print(f"Hasil akhir telah disimpan ke {file_excel_hasil_akhir}")

# 22. Tampilkan Hasil Akhir ke Layar Monitor
print("Data Hasil Akhir:")
print(data_asli[['Time', 'Device name', 'Acceleration X(g)', 'Acceleration Y(g)', 'Acceleration Z(g)',
                 'Angular velocity X(°/s)', 'Angular velocity Y(°/s)', 'Angular velocity Z(°/s)',
                 'Angle X(°)', 'Angle Y(°)', 'Angle Z(°)', 'Predicted Change']])
