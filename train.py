#train.py
import os
import numpy as np
from tensorflow.keras.models import load_model
from normalize import process_image
from plot import plot_image_2


def load_images_from_directory(directory, start=45, end=50):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(directory, filename)
            images.append(img_path)
    sorted_images = sorted(images)
    return sorted_images[start:end]

low_quality_dir = r"C:\Users\sunda\Downloads\Dataset\DIV2K_train_LR_bicubic_X4\X4"
high_quality_dir = r"C:\Users\sunda\Downloads\Dataset\DIV2K_train_HR"


low_quality_images = load_images_from_directory(low_quality_dir)
high_quality_images = load_images_from_directory(high_quality_dir)

x_train_list = []
y_train_list = []

for high_img in high_quality_images:
    base_name = os.path.basename(high_img).split('.')[0]
    
    low_img_name = f"{base_name}x4.png"
    low_img_path = os.path.join(low_quality_dir, low_img_name)
    
    if os.path.exists(low_img_path):
        processed_image = process_image(low_img_path)
        original_image = process_image(high_img)
        x_train_list.append(processed_image)
        y_train_list.append(original_image)

x_train = np.array(x_train_list)
y_train = np.array(y_train_list)

print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")

#plot_image_2(x_train[1], y_train[1])

model = load_model('srcnn_model.h5')

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=50, batch_size=5)
model.save('srcnn_model.h5')

