import os
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Load the trained model
from torchvision.models import resnet18

# Initialize Flask app
app = Flask(__name__)

# Define upload folder
UPLOAD_FOLDER = "C:\\Users\\bambp\\Desktop\\Plant Disease Detection And Fertilizer Suggestion\\Templates\\static\\uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Load the trained model
device = torch.device("cpu")
model = resnet18(pretrained=False)
num_classes = 38  # Update based on your dataset
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("best_resnet_model.pth", map_location=device))
model.to(device)
model.eval()

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Define class labels
classes = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_", "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy", "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy", "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight", "Potato___Late_blight",
    "Potato___healthy", "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy", "Tomato___Bacterial_spot", "Tomato___Early_blight",
    "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
]

# Define fertilizer suggestions
fertilizer_suggestions = {
    "Apple___Apple_scab": "Apply fungicides containing captan or copper and ensure proper pruning.",
    "Apple___Black_rot": "Use fungicides containing thiophanate-methyl or myclobutanil.",
    "Apple___Cedar_apple_rust": "Spray sulfur-based fungicides and remove infected leaves.",
    "Apple___healthy": "Maintain a balanced fertilization schedule with nitrogen, phosphorus, and potassium.",
    "Blueberry___healthy": "Use ammonium-based fertilizers and maintain acidic soil conditions (pH 4.5-5.5).",
    "Cherry_(including_sour)___Powdery_mildew": "Apply sulfur-based or potassium bicarbonate fungicides.",
    "Cherry_(including_sour)___healthy": "Use compost or balanced fertilizers rich in potassium and nitrogen.",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Apply strobilurin fungicides and ensure crop rotation.",
    "Corn_(maize)___Common_rust_": "Use fungicides with propiconazole or pyraclostrobin.",
    "Corn_(maize)___Northern_Leaf_Blight": "Apply fungicides containing azoxystrobin or mancozeb.",
    "Corn_(maize)___healthy": "Fertilize with nitrogen-rich fertilizers during the growing season.",
    "Grape___Black_rot": "Use fungicides containing myclobutanil or mancozeb and prune infected vines.",
    "Grape___Esca_(Black_Measles)": "Apply fungicides during the early growing season and improve drainage.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Use copper-based fungicides and avoid overhead irrigation.",
    "Grape___healthy": "Use balanced fertilizers with potassium, phosphorus, and magnesium.",
    "Orange___Haunglongbing_(Citrus_greening)": "Apply micronutrient sprays and systemic insecticides for psyllid control.",
    "Peach___Bacterial_spot": "Use copper-based bactericides and avoid overhead irrigation.",
    "Peach___healthy": "Fertilize with nitrogen during the early growing season.",
    "Pepper,_bell___Bacterial_spot": "Apply copper-based fungicides and avoid waterlogged conditions.",
    "Pepper,_bell___healthy": "Use balanced fertilizers rich in nitrogen and potassium.",
    "Potato___Early_blight": "Apply fungicides containing chlorothalonil or mancozeb.",
    "Potato___Late_blight": "Use fungicides with fluazinam or mancozeb and ensure good airflow.",
    "Potato___healthy": "Fertilize with balanced NPK fertilizers during tuber development.",
    "Raspberry___healthy": "Use compost or potassium-rich fertilizers to encourage healthy growth.",
    "Soybean___healthy": "Inoculate seeds with rhizobia bacteria and apply phosphorus-rich fertilizers.",
    "Squash___Powdery_mildew": "Spray sulfur-based fungicides or potassium bicarbonate solutions.",
    "Strawberry___Leaf_scorch": "Apply fungicides containing thiophanate-methyl or mancozeb.",
    "Strawberry___healthy": "Use balanced fertilizers with potassium and magnesium for fruit development.",
    "Tomato___Bacterial_spot": "Spray copper-based bactericides and remove infected plants.",
    "Tomato___Early_blight": "Use fungicides containing chlorothalonil or copper-based sprays.",
    "Tomato___Late_blight": "Apply fungicides with mancozeb or chlorothalonil.",
    "Tomato___Leaf_Mold": "Use sulfur-based fungicides and improve ventilation.",
    "Tomato___Septoria_leaf_spot": "Spray fungicides containing mancozeb or copper oxychloride.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Use miticides or neem oil for control.",
    "Tomato___Target_Spot": "Apply fungicides with azoxystrobin or mancozeb.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Control whiteflies with systemic insecticides.",
    "Tomato___Tomato_mosaic_virus": "Remove infected plants and disinfect tools.",
    "Tomato___healthy": "Use a balanced fertilizer with calcium and potassium to support growth." 
    
}
for cls in classes:
    if "healthy" in cls.lower():
        fertilizer_suggestions[cls] = "Healthy: No fertilizer needed right now."

# Function to process and predict image
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = classes[predicted.item()]
    
    suggestion = fertilizer_suggestions.get(predicted_class, "No suggestion available.")
    
    return predicted_class, suggestion

# Flask route to upload and process image
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", error="No file selected.")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="No file selected.")

        if file and file.filename.split(".")[-1].lower() in ALLOWED_EXTENSIONS:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Predict the disease
            predicted_class, suggestion = predict_image(filepath)

            return render_template("index.html", filename=filename, result=predicted_class, suggestion=suggestion)
    
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
