from normalize import process_image
from plot import plot_image_2
import numpy as np
from tensorflow.keras.models import load_model



low_quality_images = ["img/1.jpg", "img/2.jpg", "img/3.jpg","img/4.jpg","img/5.jpg"]  
high_quality_images = ["img/1h.png", "img/2h.png", "img/3h.png","img/4h.png","img/5h.png"]

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

#plot_image_2(x_train[1],y_train[1])




model = load_model('srcnn_model.h5')

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=1000, batch_size = 5)


model.save('srcnn_model.h5')
