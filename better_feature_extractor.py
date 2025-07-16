import numpy as np
import tensorflow as tf
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ['XLA_FLAGS'] = '--xla_gpu_strict_conv_algorithm_picker=false'

from keras.layers import Rescaling

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


def orthogonalize_vector(vector, existing_vectors):
    """
    Orthogonalize a vector with respect to existing vectors.
    
    Args:
        vector (tf.Tensor): The vector to orthogonalize
        existing_vectors (list): List of existing orthogonal vectors
    
    Returns:
        tf.Tensor: Orthogonalized vector
    """
    for existing_vec in existing_vectors:
        # Gram-Schmidt orthogonalization
        vector = vector - tf.reduce_sum(
            tf.multiply(
                vector, 
                existing_vec
            ) / tf.reduce_sum(
                tf.multiply(existing_vec, existing_vec)
            ) * existing_vec
        )
    return vector

def load_image(filename):
    """
    Load and preprocess an image.

    Args:
        filename (str): Path to the image file.

    Returns:
        tensor: Preprocessed image tensor.
    """
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [height, width])
    img.set_shape([height, width, 3])  # Explicitly set the shape
    return img

def load_images_by_label(path_images, path_labels, label_name):
    """
    Load images by label.

    Args:
        path_images (str): Path to the images directory.
        path_labels (str): Path to the labels file.
        label_name (str): The label name to filter images.

    Returns:
        tuple: A tuple containing datasets of positive and negative labeled images.
    """
    annotations, annotations_name = load_labels(path_labels)
    if annotations is None or annotations_name is None:
        return None, None

    try:
        label_index = annotations_name.index(label_name)
        print(f"Label index: {label_index}")
    except ValueError:
        print(f"Error: The label {label_name} was not found in the annotations.")
        return None, None
        
    label_positive_filenames = []
    label_negative_filenames = []
    
    for filename in os.listdir(path_images):
        image_path = os.path.join(path_images, filename)
        image_name = os.path.basename(image_path)
        
        # Find the label of the image
        index = np.where(annotations[:, 0] == image_name)[0]
        if len(index) == 0:
            continue
        
        # Check value of attribute at label_index
        if annotations[index, label_index] == '1':
            label_positive_filenames.append(image_path)
        else:
            label_negative_filenames.append(image_path)
    
    label_positive_dataset = tf.data.Dataset.from_tensor_slices(label_positive_filenames)
    label_positive_dataset = label_positive_dataset.map(lambda x: tf.py_function(load_image, [x], tf.float32))
    label_positive_dataset = label_positive_dataset.map(lambda x: tf.ensure_shape(x, [height, width, 3])).batch(batch_size)
    
    label_negative_dataset = tf.data.Dataset.from_tensor_slices(label_negative_filenames)
    label_negative_dataset = label_negative_dataset.map(lambda x: tf.py_function(load_image, [x], tf.float32))
    label_negative_dataset = label_negative_dataset.map(lambda x: tf.ensure_shape(x, [height, width, 3])).batch(batch_size)
    
    # Normalize the images
    normalize = Rescaling(1./255)
    label_positive_dataset = label_positive_dataset.map(lambda x: normalize(x))
    label_negative_dataset = label_negative_dataset.map(lambda x: normalize(x))
    
    return label_positive_dataset, label_negative_dataset

def extract_attribute_vector_advanced(
    model, 
    path_images, 
    path_labels, 
    label_name, 
    existing_attribute_vectors=None,
    num_samples=None,
    orthogonalize=True,
    correlation_threshold=0.3
):
    """
    Advanced attribute vector extraction with improved feature separation.

    Args:
        model (keras.Model): The trained VAE model
        path_images (str): Path to images directory
        path_labels (str): Path to labels file
        label_name (str): Attribute label to extract
        existing_attribute_vectors (list, optional): List of previously extracted attribute vectors
        num_samples (int, optional): Number of samples to use for more robust extraction
        orthogonalize (bool): Whether to orthogonalize the attribute vector
        correlation_threshold (float): Threshold for attribute correlation

    Returns:
        tf.Tensor: Refined attribute vector
    """
    # Load datasets
    attribute_dataset, non_attribute_dataset = load_images_by_label(
        path_images, path_labels, label_name
    )
    if attribute_dataset is None or non_attribute_dataset is None:
        return None

    # Sampling strategy
    if num_samples:
        attribute_dataset = attribute_dataset.take(num_samples)
        non_attribute_dataset = non_attribute_dataset.take(num_samples)

    # Encode images
    z_mean_with_attribute = []
    for batch in attribute_dataset:
        z_mean, _, _ = model.encoder.predict(batch)
        z_mean_with_attribute.append(z_mean)
    if z_mean_with_attribute:
        z_mean_with_attribute = np.concatenate(z_mean_with_attribute, axis=0)
    else:
        print(f"No images found with attribute {label_name}")
        return None

    z_mean_without_attribute = []
    for batch in non_attribute_dataset:
        z_mean, _, _ = model.encoder.predict(batch)
        z_mean_without_attribute.append(z_mean)
    if z_mean_without_attribute:
        z_mean_without_attribute = np.concatenate(z_mean_without_attribute, axis=0)
    else:
        print(f"No images found without attribute {label_name}")
        return None

    # Compute mean representations
    mean_with_attribute = tf.reduce_mean(z_mean_with_attribute, axis=0, keepdims=True)
    mean_without_attribute = tf.reduce_mean(z_mean_without_attribute, axis=0, keepdims=True)

    # Compute attribute vector
    attribute_vector = mean_with_attribute - mean_without_attribute

    # Variance reduction
    attribute_vector_std = tf.math.reduce_std(
        z_mean_with_attribute - z_mean_without_attribute, 
        axis=0
    )
    attribute_vector = attribute_vector / (attribute_vector_std + 1e-8)

    # Orthogonalization
    if orthogonalize and existing_attribute_vectors:
        attribute_vector = orthogonalize_vector(
            attribute_vector, 
            existing_attribute_vectors
        )

    # Correlation analysis and filtering
    def compute_correlation(vec1, vec2):
        """Compute cosine similarity between two vectors."""
        return tf.reduce_sum(
            tf.multiply(vec1, vec2)
        ) / (
            tf.norm(vec1) * tf.norm(vec2)
        )

    # Filter out highly correlated attributes
    if existing_attribute_vectors:
        correlations = [
            compute_correlation(attribute_vector, existing_vec)
            for existing_vec in existing_attribute_vectors
        ]
        
        if any(abs(corr) > correlation_threshold for corr in correlations):
            print(f"Warning: Attribute vector for {label_name} is highly correlated with existing vectors.")
            # Additional decorrelation could be applied here

    # Save the refined attribute vector
    np.save(f'{label_name}_attribute_vector.npy', attribute_vector.numpy())

    return attribute_vector

def apply_attribute_vector(
    model, 
    image_latent, 
    attribute_vector, 
    strength=1.0, 
    method='additive'
):
    """
    Apply an attribute vector to an image's latent representation.

    Args:
        model (keras.Model): The VAE model
        image_latent (tf.Tensor): Latent representation of the image
        attribute_vector (tf.Tensor): Attribute vector to apply
        strength (float): Strength of attribute modification
        method (str): Method of applying attribute ('additive' or 'interpolative')

    Returns:
        tf.Tensor: Modified latent representation
    """
    if method == 'additive':
        modified_latent = image_latent + strength * attribute_vector
    elif method == 'interpolative':
        modified_latent = image_latent * (1 - strength) + strength * attribute_vector
    else:
        raise ValueError("Method must be 'additive' or 'interpolative'")

    return modified_latent