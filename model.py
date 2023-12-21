from keras.layers import Input, Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, Flatten, Concatenate, Dense
from keras.models import Model

def create_multi_input_model():
    # Define input shapes
    image_input_shape = (15, 128, 128, 3)
    audio_input_shape = (20, 47, 1)

    # Image input and layers
    image_input = Input(shape=image_input_shape)
    conv3d_1 = Conv3D(32, kernel_size=(3, 3, 3), activation='relu')(image_input)
    maxpool3d_1 = MaxPooling3D(pool_size=(2, 2, 2))(conv3d_1)
    conv3d_2 = Conv3D(64, kernel_size=(3, 3, 3), activation='relu')(maxpool3d_1)
    maxpool3d_2 = MaxPooling3D(pool_size=(2, 2, 2))(conv3d_2)
    conv3d_3 = Conv3D(128, kernel_size=(2, 2, 2), activation='relu')(maxpool3d_2)
    maxpool3d_3 = MaxPooling3D(pool_size=(1, 1, 1))(conv3d_3)
    conv3d_4 = Conv3D(256, kernel_size=(1, 1, 1), activation='relu')(maxpool3d_3)
    maxpool3d_4 = MaxPooling3D(pool_size=(1, 1, 1))(conv3d_4)
    flat_image = Flatten()(maxpool3d_4)

    # Audio input and layers
    audio_input = Input(shape=audio_input_shape)
    conv2d_aud1 = Conv2D(128, kernel_size=(3, 3), activation='relu')(audio_input)
    maxpool2d_aud1 = MaxPooling2D(pool_size=(2, 2))(conv2d_aud1)
    conv2d_aud2 = Conv2D(64, kernel_size=(3, 3), activation='relu')(maxpool2d_aud1)
    maxpool2d_aud2 = MaxPooling2D(pool_size=(2, 2))(conv2d_aud2)
    conv2d_aud3 = Conv2D(32, kernel_size=(3, 3), activation='relu')(maxpool2d_aud2)
    maxpool2d_aud3 = MaxPooling2D(pool_size=(1, 1))(conv2d_aud3)
    flat_audio = Flatten()(maxpool2d_aud3)

    # Concatenate features and create output layer
    concatenated_features = Concatenate()([flat_image, flat_audio])
    output = Dense(2, activation='softmax')(concatenated_features)

    # Create and return the model
    model = Model(inputs=[image_input, audio_input], outputs=output)
    return model

