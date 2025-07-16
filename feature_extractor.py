import os
import sys
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ['XLA_FLAGS'] = '--xla_gpu_strict_conv_algorithm_picker=false'

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers import Rescaling

height = 128
width = 128
batch_size = 32

from keras.layers import Rescaling
import tensorflow as tf

# Define paths and parameters
face_images_dir = 'Dataset/all_images'
annotations_dir = 'Dataset/list_attr_celeba.txt'
height = 128
width = 128
batch_size = 32

def load_labels(path_labels):
    """
    Load labels from a file.

    Args:
        path_labels (str): Path to the labels file.

    Returns:
        tuple: A tuple containing annotations and annotation names.
    """
    annotations = []
    try:
        with open(path_labels) as f:
            lines = f.readlines()
            for line in lines[2:]:
                line = line.split()
                annotations.append(line)
    except FileNotFoundError:
        print(f"Error: The file {path_labels} was not found.")
        return None, None

    annotations = np.array(annotations)
    annotations_name = lines[1].split()[:]
    return annotations, annotations_name

# Separate filenames based on the "Smiling" attribute
def load_image(filename):
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [height, width])
    return img

def get_attribute_index(annotations_name, attribute_name):
    """
    Get the index of the attribute.

    Args:
        annotations_name (list): List of attribute names.
        attribute_name (str): Name of the attribute.

    Returns:
        int: Index of the attribute.
    """
    return annotations_name.index(attribute_name)

def extract_attribute_vector(model, image_dir, annotations_dir, attribute_name):
    """
    Extract the attribute vector from the annotations.

    Args:
        annotations (np.ndarray): Annotations array.
        attribute_name (str): Name of the attribute.

    Returns:
        np.ndarray: Attribute vector.
    """
    annotations, annotations_name = load_labels(annotations_dir)
    attribute_index = get_attribute_index(annotations_name, attribute_name)
    attribute_filename = [os.path.join(image_dir, line[0]) for line in annotations if line[attribute_index] == '1']
    non_attribute_filename = [os.path.join(image_dir, line[0]) for line in annotations if line[attribute_index] == '-1']
    
    attribute_dataset = tf.data.Dataset.from_tensor_slices(attribute_filename).map(load_image).batch(batch_size)
    non_attribute_dataset = tf.data.Dataset.from_tensor_slices(non_attribute_filename).map(load_image).batch(batch_size)
    
    normalization_layer = Rescaling(1./255)
    attribute_dataset = attribute_dataset.map(lambda x: normalization_layer(x))
    non_attribute_dataset = non_attribute_dataset.map(lambda x: normalization_layer(x))
    
    z_mean_with_attribute, z_log_var_with_attribute, z_with_attribute = model.encoder.predict(attribute_dataset)
    z_mean_without_attribute, z_log_var_without_attribute, z_without_attribute = model.encoder.predict(non_attribute_dataset)
    
    mean_encoded_images_with_smile = tf.reduce_mean(z_mean_with_attribute, axis=0, keepdims=True)
    mean_encoded_images_without_smile = tf.reduce_mean(z_mean_without_attribute, axis=0, keepdims=True)
    
    attribute_vector = mean_encoded_images_with_smile - mean_encoded_images_without_smile
    
    return attribute_vector

def extract_attribute_vector(model, encodings, annotations_file, attribute_name):
    """
    Extract the attribute vector from the encodings.

    Args:
        model: The model to use for encoding.
        encodings (np.ndarray): Encoded images.
        annotations_file (str): Path to the annotations file.
        attribute_name (str): Name of the attribute.

    Returns:
        np.ndarray: Attribute vector.
    """
    annotations, annotations_name = load_labels(annotations_file)
    attribute_index = get_attribute_index(annotations_name, attribute_name)
    
    attribute_vector = []
    for i in range(len(annotations)):
        if annotations[i][attribute_index] == '1':
            attribute_vector.append(encodings[i])
            
    without_attribute_vector = []
    for i in range(len(annotations)):
        if annotations[i][attribute_index] == '-1':
            without_attribute_vector.append(encodings[i])
            
    mean_encoded_with_attribute = np.mean(attribute_vector, axis=0)
    mean_encoded_without_attribute = np.mean(without_attribute_vector, axis=0)
    attribute_vector = mean_encoded_with_attribute - mean_encoded_without_attribute
    
    return attribute_vector

def extract_attribute_vectors_from_encodings(encodings, annotations_file):
    """
    Extract the attribute vectors from the encodings.

    Args:
        model: The model to use for encoding.
        encodings (np.ndarray): Encoded images.
        annotations_file (str): Path to the annotations file.

    Returns:
        dict: Dictionary of attribute vectors.
    """
    annotations, annotations_name = load_labels(annotations_file)
    
    attribute_vectors = {}
    for attribute_name in annotations_name:
        attribute_index = get_attribute_index(annotations_name, attribute_name)
        
        attribute_vector = []
        for i in range(len(annotations)):
            if annotations[i][attribute_index] == '1':
                attribute_vector.append(encodings[i])
                
        without_attribute_vector = []
        for i in range(len(annotations)):
            if annotations[i][attribute_index] == '-1':
                without_attribute_vector.append(encodings[i])
                
        mean_encoded_with_attribute = np.mean(attribute_vector, axis=0)
        mean_encoded_without_attribute = np.mean(without_attribute_vector, axis=0)
        attribute_vector = mean_encoded_with_attribute - mean_encoded_without_attribute
        
        attribute_vectors[attribute_name] = attribute_vector
        
    return attribute_vectors
    
    
def encode_whole_dataset(model, image_dir):
    """
    Encode the whole dataset.

    Args:
        model: The model to use for encoding.
        image_dir (str): Directory containing images.

    Returns:
        np.ndarray: Encoded dataset.
    """
    filenames = os.listdir(image_dir)
    filenames = [os.path.join(image_dir, filename) for filename in filenames]
    
    dataset = tf.data.Dataset.from_tensor_slices(filenames).map(load_image).batch(batch_size)
    
    normalization_layer = Rescaling(1./255)
    dataset = dataset.map(lambda x: normalization_layer(x))
    
    z_mean, z_log_var, z = model.encoder.predict(dataset)
    
    return z
    
