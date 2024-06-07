from tensorflow.keras import layers, models



def create_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')) 
    return model

input_shape = (256, 256, 3)
model = create_model(input_shape)

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model.summary()


model.save('srcnn_model.h5')


