from normalize import process_image
from plot import plot_image
import numpy as np


low_quality_images = ["1.jpg", "2.jpg", "3.jpg","4.jpg","5.jpg"]  
high_quality_images = ["1h.png", "2h.png", "3h.png","4h.png","5h.png"]

x_train_list = []
y_train_list = []

for i in range(len(low_quality_images)):
    processed_image =process_image(low_quality_images[i])
    original_image = process_image(high_quality_images[i])
    x_train_list.append(processed_image)
    y_train_list.append(original_image)
x_train = np.array(x_train_list)
y_train = np.array(y_train_list)

print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")

plot_image(x_train[1],y_train[1])

