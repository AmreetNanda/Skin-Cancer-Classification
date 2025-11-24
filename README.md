# Skin Cancer Classification App (Deep Learning + Flask + Docker)

The Skin Cancer Classification Web App is an interactive, user-friendly tool that detects skin diseases from uploaded images using a deep learning model. This project demonstrates how a machine learning model can be integrated with a web interface to provide an intuitive experience for users who want to identify skin
conditions quickly and accurately.

It includes:

- A complete deep learning prediction pipeline
- A Flask web interface for uploading images and getting predictions
- A fully modular ML codebase
- Optional Docker container for deployment
---

## Features

- Image Upload Form: Users can upload a skin image for classification.
- Deep Learning Classification: The application predicts the disease type based on the uploaded image.
- User-Friendly Interface: Provides prediction results with confidence levels.
- Disease Information: Displays detailed info about common skin diseases including symptoms, causes and treatments.

## Diseases Detected:
- Vascular lesions: Red/purple spots, visible blood vessels.
- Actinic keratoses: Rough, scaly patches caused by sun exposure.
- Basal cell carcinoma: Pearly bumps or open sores caused by UV damage.
- Benign keratosis-like lesions: Waxy, wart-like growths.
- Dermatofibroma: Firm, small nodules from minor skin injuries.
- Melanocytic nevi: Dark, round/oval spots (moles).
- Melanoma: Irregular dark moles, potentially cancerous.

## Technologies Used:
- Front-End: HTML, CSS, Bootstrap (optional)
- Back-End: Python (Flask framework)
- Deep Learning for image classification: 
  - Models used:
    - Convolutional Neural Networks (CNN), 
    - VGG16, 
    - DenseNet121, 
    - Inception_Resnet 
  - Optimizers used : Adam Optimizers

## Project Structure

```bash
Skin-Cancer-Classification/
│
├── app.py # Flask app
├── Skin-Cancer-Classification/ # Modular ML package
│ ├── config
│ │   ├── config.yaml
│ │   └── params.yaml
│ ├── models
│ │   ├── H5_model
│ │   │   └── models (.pkl and .h5 models files)
│ │   ├── Reports
│ │   │   └── report files in csv format
│ ├── Research (jupyter notebook files)
│ ├── Source
│ │   ├── HAM10000_images_part_1
│ │   ├── HAM10000_images_part_2
│ │   └── csv files
│ ├── templates
│ │   ├── index.html
│ │   └── result.html
│ ├── src
│ │   ├── skin_disease_classifier
│ │   │   ├── __init__.py
│ │   │   ├── components
│ │   │   │   ├── __init__.py
│ │   │   │   ├── data_ingestion.py
│ │   │   │   ├── prepare_base_model.py
│ │   │   │   ├── prepare_callbacks.py
│ │   │   │   ├── training.py
│ │   │   │   └── evaluation.py
│ │   │   ├── config
│ │   │   │   ├── __init__.py
│ │   │   │   ├── configuration.py
│ │   │   ├── constants
│ │   │   │   ├── __init__.py
│ │   │   ├── entity
│ │   │   │   ├── __init__.py
│ │   │   │   ├── config_entity.py
│ │   │   ├── pipeline
│ │   │   │   ├── __init__.py
│ │   │   │   ├── predict.py
│ │   │   │   ├── stage_01_data_ingestion.py
│ │   │   │   ├── stage_02_prepare_base_model.py
│ │   │   │   ├── stage_03_training.py
│ │   │   │   └── stage_04_evaluation.py
│ │   │   ├── utils
│ │   │   │   ├── __init__.py
│ │   │   │   ├── common.py
│ │   │   │   └── predict_utils.py
│ ├── static 
│ ├── app.py
│ ├── dvc.yaml
│ ├── template.py
│ └── main.py
├── requirements.txt # Python dependencies
├── Dockerfile # Docker container
├── run.sh # Optional to run the script
└── setup.py # Optional
```

## Installation

## 🛠 Installation (without Docker)

### 1. Clone the repo
```bash
git clone https://github.com/AmreetNanda/Skin-Cancer-Classification.git
cd Skin_Cancer_Classification
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Flask app
```bash
run app.py
```
Open in your browser:
👉 http://127.0.0.1:5000/
👉 Upload a skin image in the form 
👉 Click the "Predict Disease" button.
👉 Receive the predicted skin disease and confidence level.

## 🐳 Running with Docker (optional)
### Build the image
```bash
docker build -t Skin_Cancer_Classification .

```

### Run the container
```bash
docker run -p 8501:8501 Skin_Cancer_Classification

```
Open: 👉 http://localhost:8501

## License

[MIT](https://choosealicense.com/licenses/mit/)
## Screenshots

##### Home page
![App Screenshot](https://github.com/AmreetNanda/Skin-Cancer-Classification/blob/main/Skin_Cancer_Classification_0.png)

##### Form 
![App Screenshot](https://github.com/AmreetNanda/Skin-Cancer-Classification/blob/main/Skin_Cancer_Classification_1.png)

## Demo 1
https://github.com/user-attachments/assets/33e033df-f710-4fde-b8af-97195e38e1a9

## Demo 2
https://github.com/user-attachments/assets/4396ed1e-179f-4773-8822-de5c0331438a

