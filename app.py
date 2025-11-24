# import os
# import numpy as np
# from tensorflow.keras.models import load_model
# from PIL import Image


# class PredictionPipeline:
#     def __init__(self, filename: str):
#         self.filename = filename

#     def predict(self):
#         # Load trained model
#         model = load_model(os.path.join("models", "H5_Format", "model_cnn_adam1.h5"))

#         # Load and preprocess image using Pillow
#         img = Image.open(self.filename).convert("RGB")
#         img = img.resize((224, 224))  # must match training input size
#         img_array = np.array(img, dtype=np.float32)
#         img_array = np.expand_dims(img_array, axis=0)

#         # Predict class probabilities
#         probs = model.predict(img_array)
#         result = np.argmax(probs, axis=1)[0]
#         confidence = float(np.max(probs)) * 100

#         # Map predicted index to disease name
#         class_map = {
#             0: "Vascular lesions",
#             1: "Actinic keratoses",
#             2: "Basal cell carcinoma",
#             3: "Benign keratosis-like lesions",
#             4: "Dermatofibroma",
#             5: "Melanocytic nevi",
#             6: "Melanoma",
#         }

#         prediction = class_map.get(result, "Unknown")

#         return [{"image": prediction, "confidence": f"{confidence:.2f}%"}]

# import os
# from flask import Flask, render_template, request
# # from predict import PredictionPipeline
# from src.skin_disease_classifier.pipeline.predict import PredictionPipeline

# app = Flask(__name__)
# UPLOAD_FOLDER = os.path.join("static", "uploads")
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# @app.route("/", methods=["GET", "POST"])
# def index():
#     if request.method == "POST":
#         if "file" not in request.files:
#             return render_template("index.html", error="No file uploaded")

#         file = request.files["file"]
#         if file.filename == "":
#             return render_template("index.html", error="No file selected")

#         filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
#         file.save(filepath)

#         pipeline = PredictionPipeline(filepath)
#         result = pipeline.predict()
#         prediction = result[0]["image"]
#         confidence = result[0]["confidence"]

#         return render_template(
#             "index.html",
#             filename=file.filename,
#             prediction=prediction,
#             confidence=confidence
#         )
#     return render_template("index.html")

# if __name__ == "__main__":
#     app.run(debug=True)

import os
from flask import Flask, render_template, request
from src.skin_disease_classifier.pipeline.predict import PredictionPipeline

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":

        if "file" not in request.files:
            return render_template("index.html", error="No file uploaded")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="No file selected")

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        pipeline = PredictionPipeline(filepath)
        result = pipeline.predict()

        prediction = result[0]["image"]
        confidence_str = result[0]["confidence"]
        confidence = float(confidence_str.strip('%'))

        # 👉 FIX: Return result.html instead of index.html
        return render_template(
            "result.html",
            filename=file.filename,
            prediction=prediction,
            confidence=confidence
        )

    # GET request → show index
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
