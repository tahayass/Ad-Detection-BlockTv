import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.metrics import Recall, Accuracy

import glob

from data_loader import load_data, custom_data_generator
from model import create_model

def train_model(dataset_path, batch_size, epochs, save_model_path):
    class_folders = class_folders = glob.glob(dataset_path+'/*/*')
    video_data, spectrogram_data = load_data(class_folders)

    train_video_keys, val_video_keys = train_test_split(list(video_data.keys()), test_size=0.2, random_state=42)

    train_generator = custom_data_generator(video_data, spectrogram_data, train_video_keys, batch_size)
    val_generator = custom_data_generator(video_data, spectrogram_data, val_video_keys, batch_size)

    model = create_model()

    # Compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=[Accuracy(), Recall()])

    # Train the model
    history = model.fit(train_generator, steps_per_epoch=len(train_video_keys) // batch_size,
                        epochs=epochs, validation_data=val_generator, validation_steps=len(val_video_keys) // batch_size)

    # Save the trained model
    model.save(save_model_path)

    # Plot training history
    plot_training_history(history)

def plot_training_history(history):
    plt.figure(figsize=(12, 4))

    # Plot training & validation loss values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.savefig(r'metrics.png')
    plt.show()

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description='Train a video classification model.')
    parser.add_argument('--dataset_path', help='List of class folders for data loading.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs.')
    parser.add_argument('--save_model_path', default='trained_model.h5', help='Path to save the trained model.')

    args = parser.parse_args()

    train_model(args.dataset_path, args.batch_size, args.epochs, args.save_model_path)

