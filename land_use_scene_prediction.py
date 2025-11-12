#LAND USE SCENE CLASSIFICATION 

import pandas
import PIL
import os
import numpy as np
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications import ResNet101V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D,Flatten,Dropout
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from IPython.display import display, Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

root_dir = "/kaggle/input/landuse-scene-classification/images_train_test_val"

!mkdir -p ~/.kaggle
from google.colab import files
files.upload()   # Upload the kaggle.json file when prompted

!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d apollo2506/landuse-scene-classification --unzip -p /content/landuse_data

import os

root_dir = "/content/landuse_data"
print(os.listdir(root_dir))

root_dir = "/content/landuse_data/images_train_test_val"

import os

root_dir = "/content/landuse_data/images_train_test_val"

# Initialize dictionaries
counts = {'train': {}, 'test': {}, 'validation': {}}

def count_images_in_directory(directory):
    label_counts = {}
    total_count = 0
    for label in os.listdir(directory):
        label_dir = os.path.join(directory, label)
        if os.path.isdir(label_dir):
            image_count = len([
                f for f in os.listdir(label_dir)
                if os.path.isfile(os.path.join(label_dir, f))
            ])
            label_counts[label] = image_count
            total_count += image_count
    return label_counts, total_count

# Count images in train/test/validation
for folder in ['train', 'test', 'validation']:
    folder_path = os.path.join(root_dir, folder)
    counts[folder]['label_counts'], counts[folder]['total_count'] = count_images_in_directory(folder_path)

# Print results
for folder in ['train', 'test', 'validation']:
    print(f"Folder: {folder}")
    print(f"Total images: {counts[folder]['total_count']}")
    for label, count in counts[folder]['label_counts'].items():
        print(f"  {label}: {count}")
    print()

# =========================================================
#  MOBILENETV3LARGE MODEL
# =========================================================

base_model = MobileNetV3Large(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)  # Add a fully connected layer
predictions = Dense(21, activation='softmax')(x)  # 21 classes for the dataset

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])
model.summary()

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = "/content/landuse_data/images_train_test_val/train"
val_dir = "/content/landuse_data/images_train_test_val/validation"

train_datagen = ImageDataGenerator()
validation_datagen = ImageDataGenerator()

batch_size = 32

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(224,224), batch_size=batch_size, class_mode='sparse')
validation_generator = validation_datagen.flow_from_directory(val_dir, target_size=(224,224), batch_size=batch_size, class_mode='sparse')

model.fit(train_generator, epochs=1, validation_data=validation_generator)

model.save('Land_use_Scene_Classification_MobileNetV3Large.h5')

test_dir = '/content/landuse_data/images_train_test_val/test'

test_datagen = ImageDataGenerator()

# Set batch size
batch_size = 32

# Create the test data generator
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(224, 224), batch_size=batch_size, class_mode='sparse')

# Evaluate the model on the test data
loss, accuracy = model.evaluate(test_generator)

print("Test loss:", loss)
print("Test accuracy:", accuracy)

image_path = '/content/landuse_data/images_train_test_val/train/sparseresidential/sparseresidential_000003.png'
# Load and preprocess the input image
img = image.load_img(image_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# Perform classification
prediction = model.predict(img_array)
class_index = np.argmax(prediction)

# Define the class labels
class_labels = ['agricultural', 'airplane', 'baseballdiamond', 'beach', 'buildings',
                'chaparral', 'denseresidential', 'forest', 'freeway', 'golfcourse',
                'harbor', 'intersection', 'mediumresidential', 'mobilehomepark',
                'overpass', 'parkinglot', 'river', 'runway', 'sparseresidential',
                'storagetanks', 'tenniscourt']


# Get the predicted class label
predicted_class = class_labels[class_index]

# Display the image
plt.imshow(img)
plt.axis('off')
plt.title(f'Predicted class: {predicted_class}')
plt.show()


# =========================================================
#  MOBILENETV3SMALL MODEL
# =========================================================

base_model = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)  # Add a fully connected layer
predictions = Dense(21, activation='softmax')(x)  # 21 classes for the dataset

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])
model.summary()

train_dir = "/content/landuse_data/images_train_test_val/train"
val_dir = "/content/landuse_data/images_train_test_val/validation"

train_datagen = ImageDataGenerator()
validation_datagen = ImageDataGenerator()

batch_size = 32

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(224,224), batch_size=batch_size, class_mode='sparse')
validation_generator = validation_datagen.flow_from_directory(val_dir, target_size=(224,224), batch_size=batch_size, class_mode='sparse')

model.fit(train_generator, epochs=1, validation_data=validation_generator)

model.save('Land_use_Scene_Classification_MobileNetV3Small.h5')

test_dir = '/content/landuse_data/images_train_test_val/test'
test_datagen = ImageDataGenerator()

# Set batch size
batch_size = 32

# Create the test data generator
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(224, 224), batch_size=batch_size, class_mode='sparse')

# Evaluate the model on the test data
loss, accuracy = model.evaluate(test_generator)

print("Test loss:", loss)
print("Test accuracy:", accuracy)

image_path = '/content/landuse_data/images_train_test_val/train/agricultural/agricultural_000001.png'
# Load and preprocess the input image
img = image.load_img(image_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# Perform classification
prediction = model.predict(img_array)
class_index = np.argmax(prediction)

# Define the class labels
class_labels = ['agricultural', 'airplane', 'baseballdiamond', 'beach', 'buildings',
                'chaparral', 'denseresidential', 'forest', 'freeway', 'golfcourse',
                'harbor', 'intersection', 'mediumresidential', 'mobilehomepark',
                'overpass', 'parkinglot', 'river', 'runway', 'sparseresidential',
                'storagetanks', 'tenniscourt']


# Get the predicted class label
predicted_class = class_labels[class_index]

# Display the image
plt.imshow(img)
plt.axis('off')
plt.title(f'Predicted class: {predicted_class}')
plt.show()

# =========================================================
# 🧩 RESNET101V2 MODEL
# =========================================================

base_model = ResNet101V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)  # Add a fully connected layer
predictions = Dense(21, activation='softmax')(x)  # 21 classes for the dataset

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])
model.summary()

train_dir = "/content/landuse_data/images_train_test_val/train"
val_dir = "/content/landuse_data/images_train_test_val/validation"

train_datagen = ImageDataGenerator()
validation_datagen = ImageDataGenerator()

batch_size = 32

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(224,224), batch_size=batch_size, class_mode='sparse')
validation_generator = validation_datagen.flow_from_directory(val_dir, target_size=(224,224), batch_size=batch_size, class_mode='sparse')

model.fit(train_generator, epochs=1, validation_data=validation_generator)

model.save('Land_use_Scene_Classification_resnet101V2.h5')

test_dir = '/kaggle/input/landuse-scene-classification/images_train_test_val/test'
test_datagen = ImageDataGenerator()

# Set batch size
batch_size = 32

# Create the test data generator
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(224, 224), batch_size=batch_size, class_mode='sparse')

# Evaluate the model on the test data
loss, accuracy = model.evaluate(test_generator)

print("Test loss:", loss)
print("Test accuracy:", accuracy)

# =========================================================
#  RESNET50V2 MODEL
# =========================================================

from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.models import Model

# Load ResNet50V2 base model
base_model_resnet = ResNet50V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom layers
x = base_model_resnet.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(21, activation='softmax')(x)

resnet50v2_model = Model(inputs=base_model_resnet.input, outputs=predictions)
resnet50v2_model.compile(optimizer=Adam(learning_rate=0.0001),
                         loss=SparseCategoricalCrossentropy(),
                         metrics=['accuracy'])

resnet50v2_model.summary()

# Train the model
history_resnet = resnet50v2_model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=5
)

# Save the model
resnet50v2_model.save('Land_use_Scene_Classification_ResNet50V2.h5')

# Evaluate on test data
loss, accuracy = resnet50v2_model.evaluate(test_generator)
print("Test loss (ResNet50V2):", loss)
print("Test accuracy (ResNet50V2):", accuracy)

# =========================================================
#  HYBRID MODEL: MobileNetV3Large + ResNet50V2
# =========================================================
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Concatenate

# Define input
input_tensor = Input(shape=(224, 224, 3))

# Load pretrained bases (without top)
mobilenet_base = MobileNetV3Large(weights='imagenet', include_top=False, input_tensor=input_tensor)
resnet_base = ResNet50V2(weights='imagenet', include_top=False, input_tensor=input_tensor)

# Freeze base layers to prevent retraining heavy parts
for layer in mobilenet_base.layers:
    layer.trainable = False
for layer in resnet_base.layers:
    layer.trainable = False

# Extract features
mobilenet_features = GlobalAveragePooling2D()(mobilenet_base.output)
resnet_features = GlobalAveragePooling2D()(resnet_base.output)

# Combine both features
combined = Concatenate()([mobilenet_features, resnet_features])
x = Dense(1024, activation='relu')(combined)
x = Dropout(0.3)(x)
output = Dense(21, activation='softmax')(x)

# Build hybrid model
hybrid_model = Model(inputs=input_tensor, outputs=output)

# Compile
hybrid_model.compile(optimizer=Adam(learning_rate=0.0001),
                     loss=SparseCategoricalCrossentropy(),
                     metrics=['accuracy'])

hybrid_model.summary()

# Train hybrid model
history_hybrid = hybrid_model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=5
)

# Save model
hybrid_model.save('Land_use_Scene_Classification_HYBRID.h5')

# Evaluate hybrid model
loss, accuracy = hybrid_model.evaluate(test_generator)
print("Test loss (Hybrid):", loss)
print("Test accuracy (Hybrid):", accuracy)


# =========================================================
#  VISUALIZATION OF ACCURACY AND LOSS
# =========================================================

plt.figure(figsize=(14,6))

# Accuracy plot
plt.subplot(1,2,1)
plt.plot(history_resnet.history['accuracy'], label='ResNet50V2 Train Acc')
plt.plot(history_resnet.history['val_accuracy'], label='ResNet50V2 Val Acc')
plt.plot(history_hybrid.history['accuracy'], label='Hybrid Train Acc')
plt.plot(history_hybrid.history['val_accuracy'], label='Hybrid Val Acc')
plt.title('Model Accuracy Comparison')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1,2,2)
plt.plot(history_resnet.history['loss'], label='ResNet50V2 Train Loss')
plt.plot(history_resnet.history['val_loss'], label='ResNet50V2 Val Loss')
plt.plot(history_hybrid.history['loss'], label='Hybrid Train Loss')
plt.plot(history_hybrid.history['val_loss'], label='Hybrid Val Loss')
plt.title('Model Loss Comparison')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

