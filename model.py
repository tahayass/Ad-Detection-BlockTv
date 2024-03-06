from keras.models import Model
from keras.layers import Input, ConvLSTM2D, Conv2D, MaxPooling3D, MaxPooling2D, Flatten, Dense, Concatenate,GlobalAveragePooling2D,BatchNormalization,TimeDistributed
from keras.utils import plot_model
def create_model(input_shape_tensor=(10,320,320,3),input_shape_spectrogram=(600,1000,4)):

    # Define input layers
    input_tensor = Input(shape=input_shape_tensor, name='input_tensor')
    input_spectrogram = Input(shape=input_shape_spectrogram, name='input_spectrogram')

    # First branch: ConvLSTM2D for the 5D tensor
    x1 = ConvLSTM2D(32, kernel_size=(5, 5), padding='same', activation='relu',return_sequences=True)(input_tensor)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling3D(pool_size=(1, 3, 3), padding='same')(x1)
    x1 = ConvLSTM2D(32, kernel_size=(5, 5), padding='same', activation='relu',return_sequences=True)(input_tensor)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling3D(pool_size=(1, 3, 3), padding='same')(x1)
    x1 = ConvLSTM2D(16, kernel_size=(3, 3), activation='relu',padding='same')(x1)
    #x1 = MaxPooling2D(pool_size=(3, 3), padding='same')(x1)
    #x1 = GlobalAveragePooling2D()(x1)  # Add GlobalAveragePooling2D to reduce spatial dimensions

    # Second branch: Convolution for the spectrogram image
    x2 = Conv2D(64, kernel_size=(3, 3), activation='relu')(input_spectrogram)
    x2 = MaxPooling2D(pool_size=(5, 5))(x2)
    x2 = Conv2D(32, kernel_size=(3, 3), activation='relu')(x2)
    x2 = MaxPooling2D(pool_size=(3, 3))(x2)
    x2 = Conv2D(16, kernel_size=(3, 3), activation='relu')(x2)
    #x2 = MaxPooling2D(pool_size=(3, 3))(x2)
    #x2 = GlobalAveragePooling2D()(x2)  # Add GlobalAveragePooling2D to reduce spatial dimensions


    flat_image = Flatten()(x1)
    flat_audio = Flatten()(x2)

    # Final layers for prediction
    concatenated_features = Concatenate()([flat_image, flat_audio])
    x = Dense(128, activation='relu')(concatenated_features)
    output = Dense(2, activation='softmax', name='output')(x)

    # Create the model
    return Model(inputs=[input_tensor, input_spectrogram], outputs=output)


if __name__ == "__main__":

    # Define input shapes
    height,width,channels=320,320,3
    height_spectrogram, width_spectrogram, channels_spectrogram=600,1000,3
    input_shape_tensor = (10, height, width, channels) 
    input_shape_spectrogram = (height_spectrogram, width_spectrogram, channels_spectrogram) 
    # Create Model 
    model = create_model(input_shape_tensor,input_shape_spectrogram)
    # Display the model summary
    model.summary()
    # Save the model plot to a file
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
