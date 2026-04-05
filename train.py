import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train():
    os.makedirs('model', exist_ok=True)
    os.makedirs('dataset/low', exist_ok=True)
    os.makedirs('dataset/medium', exist_ok=True)
    os.makedirs('dataset/high', exist_ok=True)

    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_generator = datagen.flow_from_directory(
        'dataset',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    val_generator = datagen.flow_from_directory(
        'dataset',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )
    
    if train_generator.samples == 0:
        print("No images found in dataset. Please run utils/dataset_generator.py first.")
        return

    model = build_model()
    model.fit(train_generator, validation_data=val_generator, epochs=10)
    model.save('model/model.h5')
    print("Model saved to model/model.h5")

if __name__ == '__main__':
    train()
