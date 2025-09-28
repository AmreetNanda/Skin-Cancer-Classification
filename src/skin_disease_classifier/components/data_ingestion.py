import os
import glob
import pandas as pd
from PIL import Image
from sklearn.utils import resample

class DataIngestion:
    def __init__(self, config):
        self.config = config

    def load_metadata(self):
        df = pd.read_csv(self.config.metadata_csv)
        return df

    def map_image_paths(self, df):
        img_id_path_dict = {
            os.path.splitext(os.path.basename(x))[0]: x
            for x in glob.glob(os.path.join(self.config.image_dir, '*', '*.jpg'))
        }
        df['image_path'] = df['image_id'].map(img_id_path_dict.get)
        return df

    def enrich_metadata(self, df):
        lesion_type_dict = {
            'nv': 'Melanocytic nevi',
            'mel': 'Melanoma',
            'bkl': 'Benign keratosis-like lesions',
            'bcc': 'Basal cell carcinoma',
            'akiec': 'Actinic keratoses',
            'vasc': 'Vascular lesions',
            'df': 'Dermatofibroma'
        }
        df['cell_type'] = df['dx'].map(lesion_type_dict.get)
        df['cell_type_cat'] = pd.Categorical(df['cell_type'])
        df['cell_type_idx'] = df['cell_type_cat'].codes
        df['age'].fillna(df['age'].mean(), inplace=True)
        return df

    def balance_dataset(self, df, n_samples=500):
        balanced_parts = []
        for code, group in df.groupby('dx'):
            if len(group) < n_samples:
                g = resample(group, replace=True, n_samples=n_samples, random_state=42)
            else:
                g = resample(group, replace=False, n_samples=n_samples, random_state=42)
            balanced_parts.append(g)
        balanced = pd.concat(balanced_parts).reset_index(drop=True)
        return balanced

    def save(self, df, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
