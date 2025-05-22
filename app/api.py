from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io
import logging
from fashionml import Probability_model  # Импорт вашей модели

app = Flask(__name__)

# Настройка логирования
logging.basicConfig(filename='app.log', level=logging.INFO)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Проверка входных данных
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        image_file = request.files['image']
        image = Image.open(image_file.stream).convert('L').resize((28, 28))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Предсказание
        predictions = Probability_model.predict(image_array)
        predicted_label = np.argmax(predictions[0])

        # Логирование
        app.logger.info(f"Prediction successful: {predicted_label}")

        return jsonify({"class": predicted_label.item()})
    
    except Exception as e:
        app.logger.error(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)