import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import os
import itertools

def resize_images(frames_dir,size):
  images = []
  labels = []
  folders = os.listdir(frames_dir)

  image_extensions = ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff']
  
  for j in folders:
    for image_path in tqdm(itertools.chain(*(glob(os.path.join(frames_dir, f'{j}/*.{ext}')) for ext in image_extensions))):
      image = cv2.imread(image_path)
      image=cv2.resize(image, (size), interpolation= cv2.INTER_LANCZOS4)
      label_extract = image_path.split('.')[0][-1]
      label_val = int(label_extract)
      images.append(image)
      labels.append(label_val)

  return np.array(images), np.array(labels)


def segment_images_by_labels(images, labels):
    """
    Segments the images array according to consecutive sequences of labels.

    :param images: Numpy array of shape (n, 128, 128, 3)
    :param labels: Numpy array of shape (n,) with labels 0 or 1
    :return: A tuple containing:
             - Numpy array of segmented images
             - Numpy array of labels for each segment
    """
    # Find the indices where the label changes
    change_indices = np.where(labels[:-1] != labels[1:])[0] + 1

    # Include the start and end indices for segmentation
    segments_start = np.insert(change_indices, 0, 0)
    segments_end = np.append(change_indices, len(labels))

    # Create the segments and corresponding labels
    segmented_images = [images[start:end] for start, end in zip(segments_start, segments_end)]
    segment_labels = [labels[start:end] for start,end in zip(segments_start, segments_end)]

    # Convert the lists to numpy arrays
    segmented_images_array = np.array(segmented_images, dtype=object)
    segment_labels_array = np.array(segment_labels, dtype=object)

    return segmented_images_array, segment_labels_array


def segment_arrays(images, labels, m):
    """
    Segments the images and labels arrays into segments of size m, and outputs two numpy arrays.
    Discards any remainder that does not fit into these segments.

    :param images: Numpy array of shape (n, 128, 128, 3)
    :param labels: Numpy array of shape (n,) with labels 0 or 1
    :param m: Size of each segment
    :return: Two numpy arrays, one for segmented images and one for labels of each segment
    """
    n = images.shape[0]
    num_segments = n // m  # Calculate the number of full segments that fit into the array

    # Reshape images array to have segments of size m
    images_segmented = images[:num_segments * m].reshape(-1, m, 128, 128, 3)

    # Create labels array for each segment
    labels_segmented = labels[:num_segments * m].reshape(-1, m)
    # Assuming uniform labels within each segment, take the first label of each segment
    segment_labels = labels_segmented[:, 0]

    return images_segmented, segment_labels

def generate_labels_with_mixed_long_chains(size, min_chain_length, max_chain_length):
    """
    Generates a labels array with mixed long chains of 0s and 1s.

    :param size: Total size of the labels array
    :param min_chain_length: Minimum length of a chain of consecutive labels
    :param max_chain_length: Maximum length of a chain of consecutive labels
    :return: Numpy array of labels with mixed long chains of 0s and 1s
    """
    labels = []
    current_label = np.random.choice([0, 1])  # Start with a random label

    while len(labels) < size:
        chain_length = np.random.randint(min_chain_length, max_chain_length + 1)
        labels.extend([current_label] * chain_length)
        current_label = 1 - current_label  # Switch between 0 and 1
    
    # Truncate the list to the required size
    return np.array(labels[:size])






