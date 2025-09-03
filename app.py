import os
import cv2
import numpy as np
import pickle
import shutil
import sqlite3
import requests
import matplotlib.pyplot as plt
from flask import Flask, render_template, url_for, request, jsonify
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from PIL import Image
import io
import time

# ----------------------------
# Initialize Flask app
# ----------------------------
app = Flask(__name__)

# ----------------------------
# 1️⃣ Sensor Data Section (NodeMCU)
# ----------------------------
data_store = {"temperature": None, "humidity": None, "soil": None}

@app.route("/data", methods=["GET", "POST"])
def data():
    if request.method == "POST":
        # ESP8266 sends data here
        data_store["temperature"] = request.form.get("temperature")
        data_store["humidity"] = request.form.get("humidity")
        data_store["soil"] = request.form.get("soil")
        print("✅ Received:", data_store)
        return jsonify({"status": "success"}), 200

    elif request.method == "GET":
        # Browser requests latest data
        return jsonify(data_store)

# ----------------------------
# 2️⃣ User Login / Registration
# ----------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/userlog', methods=['GET', 'POST'])
def userlog():
    if request.method == 'POST':
        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']

        query = "SELECT name, password FROM user WHERE name = ? AND password = ?"
        cursor.execute(query, (name, password))  # Prevent SQL injection
        result = cursor.fetchall()

        if len(result) == 0:
            return render_template('index.html', msg='Sorry, Incorrect Credentials Provided, Try Again')
        else:
            return render_template('userlog.html')
    return render_template('index.html')

@app.route('/userreg', methods=['GET', 'POST'])
def userreg():
    if request.method == 'POST':
        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']
        
        print(name, mobile, email, password)

        command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
        cursor.execute(command)

        cursor.execute("INSERT INTO user VALUES (?, ?, ?, ?)", (name, password, mobile, email))
        connection.commit()

        return render_template('index.html', msg='Successfully Registered')
    
    return render_template('index.html')

@app.route('/userlog.html')
def userlogg():
    return render_template('userlog.html')

# ----------------------------
# 3️⃣ Graph Display
# ----------------------------
@app.route('/graph.html', methods=['GET', 'POST'])
def graph():
    images = [
        'http://127.0.0.1:5000/static/accuracy_plot.png',
        'http://127.0.0.1:5000/static/loss_plot.png',
        'http://127.0.0.1:5000/static/confusion_matrix.png',
        'http://127.0.0.1:5000/static/f1-scor.jpeg'
    ]
    content = ['Accuracy Graph', 'Loss Graph', 'Confusion Matrix', 'F1-Score']
    return render_template('graph.html', images=images, content=content)

# ----------------------------
# 4️⃣ Grape Leaf Disease Detection
# ----------------------------
@app.route('/image', methods=['GET', 'POST'])
def image():
    if request.method == 'POST':
        # Clear static/images directory
        dirPath = "static/images"
        if not os.path.exists(dirPath):
            os.makedirs(dirPath)
        fileList = os.listdir(dirPath)
        for fileName in fileList:
            os.remove(os.path.join(dirPath, fileName))
        
        # Get cloud_url if provided (for camera/cloud images)
        cloud_url = request.form.get('cloud_url', None)
        
        # Handle file upload
        fileName = None
        if 'filename' in request.files and request.files['filename']:
            # File upload
            file = request.files['filename']
            fileName = f"grape_leaf_{int(time.time() * 1000)}.jpg"
            file_path = os.path.join("static/images", fileName)
            file.save(file_path)
        else:
            return render_template('userlog.html', msg='No file provided')

        # Image processing
        image = cv2.imread(file_path)
        if image is None:
            return render_template('userlog.html', msg='Failed to load image')

        # Color conversion
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('static/gray.jpg', gray_image)
        
        # Apply Canny edge detection
        edges = cv2.Canny(image, 250, 254)
        cv2.imwrite('static/edges.jpg', edges)
        
        # Apply thresholding
        _, threshold2 = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
        cv2.imwrite('static/threshold.jpg', threshold2)
        
        # Apply sharpening
        kernel_sharpening = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(image, -1, kernel_sharpening)
        cv2.imwrite('static/sharpened.jpg', sharpened)
        
        # Load model and class names
        model = load_model('mobilenet_classifier.h5')
        with open('class_names.pkl', 'rb') as f:
            class_names = pickle.load(f)
        
        path = file_path

        # Preprocess input image
        def preprocess_input_image(path):
            img = load_img(path, target_size=(150, 150))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # Normalize
            return img_array

        # Make prediction
        def predict_single_image(path):
            input_image = preprocess_input_image(path)
            prediction = model.predict(input_image)
            predicted_class_index = np.argmax(prediction)
            predicted_class = class_names[predicted_class_index]
            confidence = prediction[0][predicted_class_index]
            return predicted_class, confidence 

        predicted_class, confidence = predict_single_image(path)
        
        # Remedies (English & Kannada)
        remedies = {
            'Black_rot': {
                "en": [
                    "Prune infected parts and dispose of them away from the vineyard.",
                    "Apply fungicides like Mancozeb or Copper-based sprays during the growing season.",
                    "Ensure proper airflow by maintaining spacing between vines."
                ],
                "kn": [
                    "ಸೋಂಕಿತ ಭಾಗಗಳನ್ನು ಕತ್ತರಿಸಿ ಮತ್ತು ದ್ರಾಕ್ಷಿತೋಟದಿಂದ ದೂರವಿಡಿ.",
                    "ಬೆಳವಣಿಗೆಯ ಕಾಲದಲ್ಲಿ ಮ್ಯಾಂಕೊಝೆಬ್ ಅಥವಾ ತಾಮ್ರ ಆಧಾರಿತ ಸಿಂಪಡಣೆಗಳನ್ನು ಬಳಸಿ.",
                    "ಗಾಳಿಯ ಹರಿವನ್ನು ಖಚಿತಪಡಿಸಿಕೊಳ್ಳಲು ಗಿಡಗಳ ನಡುವೆ ಸೂಕ್ತ ಅಂತರವನ್ನು ಕಾಯ್ದಿರಿಸಿ."
                ],
                "label_en": "Black_rot",
                "label_kn": "ಕಪ್ಪು ಕೊಳೆ"
            },
            'Esca_(Black_Measles)': {
                "en": [
                    "Remove and destroy affected wood during winter pruning.",
                    "Avoid water stress by ensuring proper irrigation practices.",
                    "Apply sodium arsenite sprays where permitted or consult local guidelines."
                ],
                "kn": [
                    "ಚಳಿಗಾಲದ ಕತ್ತರಿಕೆಯ ಸಮಯದಲ್ಲಿ ಸೋಂಕಿತ ಮರವನ್ನು ತೆಗೆದುಹಾಕಿ ಮತ್ತು ನಾಶಪಡಿಸಿ.",
                    "ಸರಿಯಾದ ನೀರಾವರಿ ಅಭ್ಯಾಸಗಳಿಂದ ನೀರಿನ ಒತ್ತಡವನ್ನು ತಪ್ಪಿಸಿ.",
                    "ಅನುಮತಿಯಿರುವ ಸ್ಥಳಗಳಲ್ಲಿ ಸೋಡಿಯಂ ಆರ್ಸೆನೈಟ್ ಸಿಂಪಡಣೆಗಳನ್ನು ಬಳಸಿ."
                ],
                "label_en": "Esca_(Black_Measles)",
                "label_kn": "ಎಸ್ಕಾ (ಕಪ್ಪು ಕಲೆಗಳು)"
            },
            'healthy': {
                "en": [
                    "Maintain regular pruning and balanced fertilization.",
                    "Provide consistent irrigation and monitor for early disease signs.",
                    "Encourage beneficial insects for natural pest control."
                ],
                "kn": [
                    "ನಿಯಮಿತ ಕತ್ತರಿಕೆ ಮತ್ತು ಸಮತೋಲಿತ ಗೊಬ್ಬರವನ್ನು ಕಾಯ್ದಿರಿಸಿ.",
                    "ನಿರಂತರ ನೀರಾವರಿಯನ್ನು ಒದಗಿಸಿ ಮತ್ತು ಆರಂಭಿಕ ರೋಗ ಲಕ್ಷಣಗಳಿಗಾಗಿ ಮೇಲ್ವಿಚಾರಣೆ ಮಾಡಿ.",
                    "ನೈಸರ್ಗಿಕ ಕೀಟ ನಿಯಂತ್ರಣಕ್ಕಾಗಿ ಲಾಭದಾಯಕ ಕೀಟಗಳನ್ನು ಪ್ರೋತ್ಸಾಹಿಸಿ."
                ],
                "label_en": "healthy",
                "label_kn": "ಆರೋಗ್ಯವಾಗಿದೆ"
            },
            'Leaf_blight_(Isariopsis_Leaf_Spot)': {
                "en": [
                    "Remove and burn infected leaves to prevent spread.",
                    "Spray Bordeaux mixture or Chlorothalonil fungicides as per recommendations.",
                    "Improve vineyard hygiene by clearing fallen debris and weeds."
                ],
                "kn": [
                    "ಸೋಂಕಿತ ಎಲೆಗಳನ್ನು ತೆಗೆದುಹಾಕಿ ಮತ್ತು ಒಡ್ಡುವಿಕೆಯನ್ನು ತಡೆಗಟ್ಟಲು ಸುಡಿ.",
                    "ಕೃಷಿ ಶಿಫಾರಸುಗಳಿಗೆ ಅನುಗುಣವಾಗಿ ಬೋರ್ಡೊ ಮಿಶ್ರಣ ಅಥವಾ ಕ್ಲೋರೋಥಲೋನಿಲ್ ಸಿಂಪಡಿಸಿ.",
                    "ಬಿದ್ದಿರುವ ಶಿಲಾಖಂಡಗಳು ಮತ್ತು ಕಳೆಗಳನ್ನು ತೆರವುಗೊಳಿಸಿ."
                ],
                "label_en": "Leaf_blight_(Isariopsis_Leaf_Spot)",
                "label_kn": "ಎಲೆಯ ಕೊಳೆ (ಇಸಾರಿಯೋಪ್ಸಿಸ್ ಎಲೆಯ ಕಲೆ)"
            }
        }

        if predicted_class in remedies:
            r = remedies[predicted_class]
            str_label, str_label_kannada = r["label_en"], r["label_kn"]
            rem, rem1, rem_kannada, rem1_kannada = "Remedies for Grape Leaves", r["en"], "ದ್ರಾಕ್ಷಿಯ ಎಲೆಗಳಿಗೆ ಪರಿಹಾರಗಳು", r["kn"]
        else:
            str_label, str_label_kannada = predicted_class, predicted_class
            rem, rem1, rem_kannada, rem1_kannada = "No Remedies Found", [], "ಪರಿಹಾರ ಸಿಗಲಿಲ್ಲ", []

        accuracy = f"The predicted image is {str_label} with a confidence of {confidence:.2%}"
        accuracy_kannada = f"ಪ್ರಿಡಿಕ್ಟೆಡ್ ಚಿತ್ರವು {str_label_kannada} ಆಗಿದೆ, ವಿಶ್ವಾಸದ ಮಟ್ಟ {confidence:.2%}"

        return render_template(
            'results.html',
            status=str_label,
            accuracy=accuracy,
            Remedies=rem,
            Remedies1=rem1,
            status_kannada=str_label_kannada,
            accuracy_kannada=accuracy_kannada,
            Remedies_kannada=rem_kannada,
            Remedies1_kannada=rem1_kannada,
            ImageDisplay=f"/static/images/{fileName}",
            ImageDisplay1="/static/gray.jpg",
            ImageDisplay2="/static/edges.jpg",
            ImageDisplay3="/static/threshold.jpg",
            ImageDisplay4="/static/sharpened.jpg",
            cloud_url=cloud_url
        )
    
    return render_template('userlog.html')

# ----------------------------
# Logout
# ----------------------------
@app.route('/logout')
def logout():
    return render_template('index.html')

# ----------------------------
# Run App
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)