# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization

def build_cnn(input_shape=(120, 160, 3), num_classes=7):
    model = Sequential([
        Conv2D(96, (11,11), strides=(4,4), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPool2D(pool_size=(3,3), strides=(2,2)),

        Conv2D(256, (5,5), activation='relu', padding="same"),
        BatchNormalization(),
        MaxPool2D(pool_size=(3,3), strides=(2,2)),

        Conv2D(384, (3,3), activation='relu', padding="same"),
        BatchNormalization(),

        Conv2D(384, (1,1), activation='relu', padding="same"),
        BatchNormalization(),

        Conv2D(256, (1,1), activation='relu', padding="same"),
        BatchNormalization(),
        MaxPool2D(pool_size=(3,3), strides=(2,2)),

        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model
