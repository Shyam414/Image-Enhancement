import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, Add
from tensorflow.keras.models import Model

def VDSR():
    input_img = Input(shape=(256, 256, 3))  
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    
    for _ in range(18):
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    
    x = Conv2D(3, (3, 3), padding='same')(x)
    
    # Residual connection
    output_img = Add()([x, input_img])
    
    model = Model(inputs=input_img, outputs=output_img)
    return model

model = VDSR()
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model.summary()

#model.save('srcnn_model.h5')