import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

class Trainer:
    def __init__(self, model, callbacks, config):
        self.model = model
        self.callbacks = callbacks
        self.config = config

    def preprocess(self, df):
        X = np.stack(df['image'].values)
        y = df['cell_type_idx'].values

        train_mean = X.mean(axis=(0,1,2))
        train_std = X.std(axis=(0,1,2)) + 1e-7
        X = (X - train_mean) / train_std

        y_cat = to_categorical(y, num_classes=len(np.unique(y)))
        return train_test_split(X, y_cat, test_size=0.2, random_state=123, stratify=y)

    def train(self, df):
        x_train, x_test, y_train, y_test = self.preprocess(df)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.13, random_state=2)

        train_datagen = ImageDataGenerator(rotation_range=20,
                                           width_shift_range=0.2,
                                           height_shift_range=0.2,
                                           shear_range=0.1,
                                           zoom_range=0.1,
                                           horizontal_flip=True)

        steps = max(1, x_train.shape[0] // self.config.batch_size)
        history = self.model.fit(
            train_datagen.flow(x_train, y_train, batch_size=self.config.batch_size),
            epochs=self.config.epochs,
            steps_per_epoch=steps,
            validation_data=(x_val, y_val),
            callbacks=self.callbacks
        )

        os.makedirs(self.config.report_dir, exist_ok=True)
        import pandas as pd
        pd.DataFrame(history.history).to_csv(os.path.join(self.config.report_dir, "training_history.csv"), index=False)

        return self.model, (x_test, y_test)
