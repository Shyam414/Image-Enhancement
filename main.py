from normalize import process_image
from plot import plot_image
import numpy as np
from tensorflow.keras.models import load_model



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

model = load_model('srcnn_model.h5')
model.fit(x_train, y_train, epochs=10, batch_size=5)
model.save('srcnn_model.h5')

test_image_path = "2.jpg"
test_image = load_image(test_image_path)


processed_test_image = process_image(test_image_path)
x_test = np.expand_dims(processed_test_image, axis=0)


enhanced_image = model.predict(x_test)

plot_image(x_test,enhanced_image)
