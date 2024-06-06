#test.py
from normalize import process_image
from plot import plot_image_3
import numpy as np
from tensorflow.keras.models import load_model


test_image_path = "img/1.jpg"
final="img/1h.png"


processed_test_image = process_image(test_image_path)
pro_final_img=process_image(final)
x_test = np.expand_dims(processed_test_image, axis=0)

model = load_model('srcnn_model.h5')


enhanced_image = model.predict(x_test)

plot_image_3(x_test[0], enhanced_image[0],pro_final_img)
