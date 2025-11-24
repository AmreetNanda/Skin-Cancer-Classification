import os
import numpy as np
from keras.models import load_model
from PIL import Image

class PredictionPipeline:
    def __init__(self, filename: str):
        self.filename = filename

    def predict(self):
        model_path = os.path.join("models", "H5_Format", "model_cnn_adam1.h5")
        model = load_model(model_path)

        # Load image
        img = Image.open(self.filename).convert("RGB")
        # img = img.resize((224, 224))
        img = img.resize((160, 120))
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        probs = model.predict(img_array)
        result = np.argmax(probs, axis=1)[0]
        confidence = float(np.max(probs)) * 100

        class_map = {
            0: "Vascular lesions",
            1: "Actinic keratoses",
            2: "Basal cell carcinoma",
            3: "Benign keratosis-like lesions",
            4: "Dermatofibroma",
            5: "Melanocytic nevi",
            6: "Melanoma",
        }

        prediction = class_map.get(result, "Unknown")
        return [{"image": prediction, "confidence": f"{confidence:.2f}%"}]
