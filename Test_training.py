import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Update the number of classes
num_classes = 216

# Update the dataset paths for training, testing, and validation
main_directory = "D:/train"  # Path to your training data
save_directory = "D:\Experiment\FInal" # Directory to save the trained model
test_directory = "D:/test"  # Path to your testing data

# Define constants
batch_size = 32
image_size = (128, 128)  # Update to match the cropped image size
random_seed = 42

# Define a function to load and preprocess the dataset
def load_and_preprocess_dataset(main_directory, image_size, batch_size):
    # Use ImageDataGenerator for data augmentation and preprocessing
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,  # 20% of the data will be used for validation
        horizontal_flip=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        shear_range=0.2
    )

    train_generator = datagen.flow_from_directory(
        main_directory,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        seed=random_seed
    )

    val_generator = datagen.flow_from_directory(
        main_directory,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        seed=random_seed
    )

    return train_generator, val_generator

# Load and preprocess the dataset with validation split
train_generator, val_generator = load_and_preprocess_dataset(main_directory, image_size, batch_size)

# Load the test data
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_datagen.flow_from_directory(
    test_directory,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    seed=random_seed
)

# Build the model using MobileNetV2 as the base model
base_model = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)  # Additional dense layer
x = Dense(512, activation='relu')(x)  # Additional dense layer
x = Dense(256, activation='relu')(x)  # Additional dense layer
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=15,  # Increased number of epochs
    validation_data=val_generator
)

# Save the model as a .h5 file
model.save(os.path.join(save_directory, 'medicinal_plant_model.h5'))

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(test_generator)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

print("Model training and evaluation completed.")
