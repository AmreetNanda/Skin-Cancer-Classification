# import numpy as np
# # from tensorflow.keras.models import load_model
# # from tensorflow.keras.preprocessing import image
# if not hasattr(np, "object"):
#     np.object = object
# if not hasattr(np, "bool"):
#     np.bool = bool
# if not hasattr(np, "int"):
#     np.int = int
# if not hasattr(np, "float"):
#     np.float = float

# from keras.models import load_model
# from keras.preprocessing import image
# from PIL import Image
# import os

# class PredictionPipeline:
#     def __init__(self, filename):
#         self.filename = filename


#     def predict(self):

#         #load model
#         model = load_model(os.path.join("models","H5_Format", "model_cnn_adam1.h5"))

#         # imagename = self.filename
#         # test_image = image.load_img(imagename, target_size = (224,224))
#         # test_image = image.img_to_array(test_image)
#         # test_image = np.expand_dims(test_image, axis = 0)
#         # result = np.argmax(model.predict(test_image), axis=1)
#         # print(result)

#         img = Image.open(self.filename).convert("RGB")
#         img = img.resize((224, 224))
#         img_array = np.array(img)
#         img_array = np.expand_dims(img_array, axis=0)

#         result = np.argmax(model.predict(img_array), axis=1)

#         if result[0] == 1:
#             prediction = 'Actinic keratoses'
#             return [{ "image" : prediction}]
#         elif result[0] == 2:
#             prediction = 'Basal cell carcinoma'
#             return [{ "image" : prediction}]
#         elif result[0] == 3:
#             prediction = 'Benign keratosis-like lesions'
#             return [{ "image" : prediction}]
#         elif result[0] == 4:
#             prediction = 'Dermatofibroma'
#             return [{ "image" : prediction}]
#         elif result[0] == 5:
#             prediction = 'Melanocytic nevi'
#             return [{ "image" : prediction}]
#         elif result[0] == 6:
#             prediction = 'Melanoma'
#             return [{ "image" : prediction}]
#         else:
#             prediction = 'Vascular lesions'
#             return [{ "image" : prediction}]

        
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
