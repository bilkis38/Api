from flask import Flask, request, jsonify
import pandas as pd
from sklearn.neural_network import MLPClassifier
from joblib import load

app = Flask(__name__)

# Load the trained model
model = load('model_ann.joblib')

@app.route('/prediksi', methods=['POST'])
def prediksi():
    if request.method == 'POST':
        try:
            # Dapatkan data JSON dari permintaan
            data = request.get_json()

            # Persiapkan data masukan untuk prediksi
            input_data = pd.DataFrame(data['fitur'], columns=['Acceleration X(g)', 'Acceleration Y(g)', 'Acceleration Z(g)',
                                                               'Angular velocity X(°/s)', 'Angular velocity Y(°/s)', 'Angular velocity Z(°/s)',
                                                               'Angle X(°)', 'Angle Y(°)', 'Angle Z(°)'])

            # Lakukan prediksi
            prediksi = model.predict(input_data)

            # Persiapkan respons
            respons = {
                'prediksi': prediksi.tolist(),
            }

            return jsonify(respons)

        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return "Metode tidak diizinkan!", 405  # 405 adalah kode status untuk "Method Not Allowed"

if __name__ == '__main__':
    app.run(debug=True)
