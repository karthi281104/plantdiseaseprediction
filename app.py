from flask import Flask, request, render_template, url_for
import os
import numpy as np
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing import image
import google.generativeai as genai
import markdown
from bs4 import BeautifulSoup
from flask import jsonify


app = Flask(__name__)

model = load_model('D:\harni mini project\Plant-Disease-Detection-System\models\my_model4.h5')

genai.configure(api_key="AIzaSyBATLWCzIEAfdJRsQuZqbBXf2pJDWMx310")

class_mapping = {
    0: "Apple scab",
    1: "Apple Black rot",
    2: "Cedar Apple rust",
    3: "Apple healthy",
    4: "Blueberry healthy",
    5: "Cherry Powdery mildew",
    6: "Cherry healthy",
    7: "Corn (maize) Cercospora leaf spot Gray leaf spot",
    8: "Corn (maize) Common rust",
    9: "Corn (maize) Northern Leaf Blight",
    10: "Corn (maize) healthy",
    11: "Grape Black rot",
    12: "Grape Esca (Black Measles)",
    13: "Grape Leaf blight (Isariopsis Leaf Spot)",
    14: "Grape healthy",
    15: "Orange Haunglongbing (Citrus greening)",
    16: "Peach Bacterial spot",
    17: "Peach healthy",
    18: "Pepper, bell Bacterial spot",
    19: "Pepper, bell healthy",
    20: "Potato Early blight",
    21: "Potato Late blight",
    22: "Potato healthy",
    23: "Raspberry healthy",
    24: "Soybean healthy",
    25: "Squash Powdery mildew",
    26: "Strawberry Leaf scorch",
    27: "Strawberry healthy",
    28: "Tomato Bacterial spot",
    29: "Tomato Early blight",
    30: "Tomato Late blight",
    31: "Tomato Leaf Mold",
    32: "Tomato Septoria leaf spot",
    33: "Tomato Spider mites Two spotted spider mite",
    34: "Tomato Target Spot",
    35: "Tomato Tomato Yellow Leaf Curl Virus",
    36: "Tomato Tomato mosaic virus",
    37: "Tomato healthy"
}

@app.route('/', methods=['GET'])
def home():
    if request.method == 'POST':
        if 'file' in request.files:
            return predict_single()
        elif 'folder' in request.files:
            return predict_folder()
    return render_template('homepage.html')

@app.route('/homepage')
def homepage():
    return render_template('homepage.html')

@app.route('/aboutus')
def about_us():
    return render_template('aboutus.html')

@app.route('/contactus')
def _us():
    return render_template('contactus.html')

@app.route('/single_image_prediction', methods=['POST'])
def single_image_prediction():
    if 'file' in request.files:
        # Single image prediction
        return predict_single()
    else:
        return render_template('homepage.html', prediction='No file selected for single image prediction')

@app.route('/folder_prediction', methods=['POST'])
def folder_prediction():
    if 'folder' in request.files:
        # Folder prediction
        return predict_folder()
    else:
        return render_template('homepage.html', prediction='No folder selected for folder prediction')
@app.route('/greenbot_ask', methods=['POST'])
def greenbot_ask():
    data = request.get_json()
    question = data.get('question', '').lower()

    plant_keywords = ["plant", "growth", "water", "sunlight", "fertilizer", "soil", "leaf", "disease", "pruning", "harvest"]
    is_plant_related = any(word in question for word in plant_keywords)

    if is_plant_related:
        prompt = f"You're GreenBot, an expert on plants 🌱. Answer concisely about: {question}"
    else:
        prompt = "Tell me a fun or interesting random fact about plants."

    try:
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        response = gemini_model.generate_content(prompt)
        return jsonify({"response": response.text.strip()})
    except Exception as e:
        return jsonify({"response": f"Oops! Couldn't get an answer. {str(e)}"})

def predict_single():
    if 'file' not in request.files:
        return render_template('homepage.html', prediction='There is no file in form!')
    file = request.files['file']
    if file.filename == '':
        return render_template('homepage.html', prediction='No selected file')
    if file:
        filepath = 'D:/harni mini project/uploads' + file.filename
        file.save(filepath)
        
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array_expanded_dims = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array_expanded_dims)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_class_name = class_mapping.get(predicted_class_index, "Unknown class")

        image_url = url_for('static', filename='uploads/' + file.filename)
        remedy_text = get_remedy_info(predicted_class_name)
        return render_template('homepage.html', prediction=predicted_class_name, image_url=image_url,remedy=remedy_text)

def predict_folder():
    if 'folder' not in request.files:
        return render_template('homepage.html', prediction='There is no folder in form!')
    folder = request.files.getlist('folder')
    if not folder:
        return render_template('homepage.html', prediction='No selected folder')
    
    predictions = []
    for file in folder:
        folder_name = os.path.basename(os.path.dirname(file.filename))
        
        folder_path = os.path.join('uploads', folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        filepath = os.path.join(folder_path, os.path.basename(file.filename))
        file.save(filepath)
        
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array_expanded_dims = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array_expanded_dims)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_class_name = class_mapping.get(predicted_class_index, "Unknown class")
        
        print("Predicted class name:", predicted_class_name)  # Debug print
        
        image_url = url_for('static', filename=os.path.join('uploads', folder_name, os.path.basename(file.filename)))
        
        predictions.append((predicted_class_name, image_url))

    print("Predictions:", predictions)  # Debug print

    return render_template('homepage.html', predictions=predictions)


genai.configure(api_key="AIzaSyBATLWCzIEAfdJRsQuZqbBXf2pJDWMx310")  # Replace with your key

def get_remedy_info(disease_name):
    prompt = f"Give a concise treatment plan and prevention tips for the plant disease '{disease_name}'. Include key symptoms and best practices for control. I want the response as a single short paragraph consisting of treatment and dosage information."

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        raw_text = response.text.strip()

        html_text = markdown.markdown(raw_text)
        soup = BeautifulSoup(html_text, "html.parser")
        clean_text = " ".join([p.get_text() for p in soup.find_all("p")])

        # ✅ Print remedy info in the terminal
        print(f"\n🩺 Remedy info for '{disease_name}':\n{clean_text}\n")

        return clean_text

    except Exception as e:
        error_msg = f"⚠️ Unable to fetch remedy: {str(e)}"
        print(error_msg)  # ✅ Print error to terminal
        return error_msg


if __name__ == '__main__':
    app.run(debug=True)
