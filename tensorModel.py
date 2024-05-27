def SRCNN():
    input_img = Input(shape=(256, 256, 3))  # Assuming RGB images
    x = Conv2D(64, (9, 9), activation='relu', padding='same')(input_img)
    x = Conv2D(32, (1, 1), activation='relu', padding='same')(x)
    output_img = Conv2D(3, (5, 5), activation='linear', padding='same')(x)

    model = Model(inputs=input_img, outputs=output_img)
    return model

model = SRCNN()
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()
model.save('srcnn_model.h5')