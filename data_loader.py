import os
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def load_data(class_folders):
    video_data = {}
    spectrogram_data = {}
    
    for class_folder in class_folders:
        i=0
        video_files = [f for f in os.listdir(class_folder) if f.endswith('.npy')]
        spectrogram_files = [f for f in os.listdir(class_folder) if f.endswith('.png')]

        for video_file in tqdm(video_files):
            if (i%12)==0:
                video_path = os.path.join(class_folder, video_file)
                video_data[video_file] = np.load(video_path)
            i=i+1
        i=0
        for spectrogram_file in spectrogram_files:
            if (i%12)==0:
                spectrogram_path = os.path.join(class_folder, spectrogram_file)
                spectrogram_data[spectrogram_file] = np.array(Image.open(spectrogram_path))
            i=i+1

    return video_data, spectrogram_data

def custom_data_generator(video_data, spectrogram_data, keys, batch_size):
    video_keys = list(keys)
    while True:
        try:
            batch_keys = np.random.choice(video_keys, size=batch_size)
            batch_video_data = [video_data[key] for key in batch_keys]
            batch_spectrogram_data = [spectrogram_data[key.replace('video_frames', 'spectrogram').replace('.npy','.png')] for key in batch_keys]

            labels = [int(not('intermediate' in key)) for key in batch_keys]  # Assuming 'class1' in the folder name indicates class 1
        except:
            continue
        yield [np.array(batch_video_data), np.array(batch_spectrogram_data)], to_categorical(labels, num_classes=2)
if __name__=="__main__":
    # Example usage
    class_folders = [r'D:\ad_data\output_folder_ad',r'D:\ad_data\output_folder_no_ad']
    batch_size = 1

    video_data, spectrogram_data = load_data(class_folders)

    train_video_keys, val_video_keys = train_test_split(list(video_data.keys()), test_size=0.2, random_state=42)
    print(len(train_video_keys))
    print(len(val_video_keys))
    train_generator = custom_data_generator(video_data, spectrogram_data, train_video_keys, batch_size)
    val_generator = custom_data_generator(video_data, spectrogram_data, val_video_keys, batch_size)
    for _ in range(10):
        print(next(train_generator)[0][1].shape)
    # Use train_generator and val_generator as inputs to model.fit() for training your model
