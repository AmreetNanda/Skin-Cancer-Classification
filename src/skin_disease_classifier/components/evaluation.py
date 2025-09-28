import numpy as np
import pandas as pd
import os
from sklearn.metrics import classification_report
# from tensorflow.keras.models import load_model
from keras.models import load_model

class Evaluator:
    def __init__(self, model_path, report_dir):
        self.model_path = model_path
        self.report_dir = report_dir
        os.makedirs(report_dir, exist_ok=True)

    def evaluate(self, x_test, y_test, categories):
        model = load_model(self.model_path)
        y_pred_prob = model.predict(x_test)
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_true = np.argmax(y_test, axis=1)

        report = classification_report(y_true, y_pred, target_names=categories, output_dict=True)
        df = pd.DataFrame(report).transpose()
        df.to_csv(os.path.join(self.report_dir, "classification_report.csv"))
        return df
