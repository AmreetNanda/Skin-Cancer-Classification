import os
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
# from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

def get_callbacks(save_dir):
    os.makedirs(save_dir, exist_ok=True)

    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', patience=4, factor=0.5, min_lr=1e-7, verbose=1)
    checkpoint = ModelCheckpoint(os.path.join(save_dir, "best_model.h5"),
                                 monitor='val_accuracy', save_best_only=True, verbose=1)
    earlystop = EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1)

    return [reduce_lr, checkpoint, earlystop]
