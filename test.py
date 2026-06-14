import tensorflow as tf
import cv2
import os
import numpy as np
import pickle

IMG_SIZE = 64

test_dir = r"dataset/asl_alphabet_test"

# ---------------------------
# Load model
# ---------------------------
model = tf.keras.models.load_model("model/asl_model.h5")

# ---------------------------
# Load labels
# ---------------------------
with open("model/labels.pkl", "rb") as f:
    labels = pickle.load(f)

labels = {v: k for k, v in labels.items()}

# ---------------------------
# Collect all images from subfolders
# ---------------------------
print("\n🚀 Scanning test dataset...\n")

correct = 0
total = 0

# loop through A-Z folders
for class_folder in os.listdir(test_dir):

    class_path = os.path.join(test_dir, class_folder)

    if not os.path.isdir(class_path):
        continue

    for img_file in os.listdir(class_path):

        if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(class_path, img_file)

        # Read image
        img = cv2.imread(img_path)

        if img is None:
            continue

        # Preprocess
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        # Prediction
        pred = model.predict(img, verbose=0)
        class_id = np.argmax(pred)
        predicted_label = labels[class_id]

        # True label = folder name (A, B, C...)
        true_label = class_folder

        print(f"📷 {class_folder}/{img_file} → Predicted: {predicted_label}")

        if predicted_label.lower() == true_label.lower():
            correct += 1

        total += 1

# ---------------------------
# Final result
# ---------------------------
print("\n======================")
if total > 0:
    accuracy = (correct / total) * 100
    print(f"✅ Test Accuracy: {accuracy:.2f}%")
else:
    print("❌ No images found in dataset!")

print("======================")