import os
import sys
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ['XLA_FLAGS'] = '--xla_gpu_strict_conv_algorithm_picker=false'

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import ops
from keras import layers


from PIL import Image
import matplotlib.pyplot as plt

from keras import utils, models
from keras.layers import Rescaling
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input


print('keras: ', keras.__version__)
print('tensorflow: ',tf.__version__)
print('python: ',sys.version)
print(tf.version.GIT_VERSION, tf.version.VERSION)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#####CONFIGS#####
Train = True
epochs = 100
latent_dim = 512
BATCH_SIZE_PER_REPLICA = 64
###################


strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))


BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a face."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = keras.random.SeedGenerator(280602)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = keras.random.normal(shape=(batch, dim), seed=self.seed_generator)
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon
    
def create_encoder(input_shape=(128, 128, 3), latent_dim=512):
    inputs = keras.Input(shape=input_shape)
    
    
    x = layers.Conv2D(64, (4,4), activation='leaky_relu', strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv2D(128, (4,4), activation='leaky_relu', strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv2D(256, (4,4), activation='leaky_relu', strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv2D(512, (4,4), activation='leaky_relu', strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Flatten()(x)
    
    # Add dense layers before final latent space
    x = layers.Dense(512, activation='leaky_relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    z = Sampling()([z_mean, z_log_var])
    
    return keras.Model(inputs, [z_mean, z_log_var, z], name='improved_encoder')


def create_decoder(latent_dim=512):
    latent_inputs = keras.Input(shape=(latent_dim,))
    
    x = layers.Dense(8*8*512, activation='leaky_relu')(latent_inputs)
    x = layers.Reshape((8, 8, 512))(x)
    
    x = layers.Conv2DTranspose(256, (4,4), activation='leaky_relu', strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2DTranspose(128, (4,4), activation='leaky_relu', strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2DTranspose(64, (4,4), activation='leaky_relu', strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2DTranspose(32, (4,4), activation='leaky_relu', strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    decoder_outputs = layers.Conv2DTranspose(3, (4,4), activation='sigmoid', padding='same')(x)
    
    return keras.Model(latent_inputs, decoder_outputs, name='improved_decoder')


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.lambda_reconstruction = 0.1
        self.lambda_perceptual = 0.1
        self.lambda_kl = 1
        
        # Initialize VGG19 for feature extraction (use pre-trained weights)
        vgg = VGG19(include_top=False, weights='imagenet', input_shape=(None, None, 3))
        
        # Choose intermediate layers for feature comparison
        self.feature_layers = [
            'block1_conv2',  # Low-level features
            'block2_conv2',  # Mid-level features
            'block3_conv2',  # Higher-level features
            'block4_conv2'   # Very high-level features
        ]
        
        # Create a model that outputs features from these layers
        self.feature_extractor = keras.Model(
            inputs=vgg.input, 
            outputs=[vgg.get_layer(name).output for name in self.feature_layers]
        )
        
        # Freeze the VGG19 weights
        self.feature_extractor.trainable = False
        
        # Tracking metrics
        self.total_loss_tracker = keras.metrics.Mean(name='total_loss')
        self.reconstruction_loss_tracker = keras.metrics.Mean(name='reconstruction_loss')
        self.perceptual_loss_tracker = keras.metrics.Mean(name='perceptual_loss')
        self.kl_loss_tracker = keras.metrics.Mean(name='kl_loss')
        
        # Add validation metrics
        self.val_total_loss_tracker = keras.metrics.Mean(name='val_total_loss')
        self.val_reconstruction_loss_tracker = keras.metrics.Mean(name='val_reconstruction_loss')
        self.val_perceptual_loss_tracker = keras.metrics.Mean(name='val_perceptual_loss')
        self.val_kl_loss_tracker = keras.metrics.Mean(name='val_kl_loss')
    
    def call(self, inputs):
        # Implement the forward pass
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return reconstruction
    
    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.perceptual_loss_tracker,
            self.kl_loss_tracker,
            self.val_total_loss_tracker,
            self.val_reconstruction_loss_tracker,
            self.val_perceptual_loss_tracker,
            self.val_kl_loss_tracker
        ]
    
    def compute_perceptual_loss(self, original, reconstructed):
        # Preprocess images for VGG19 (ensure 3 channels and correct scaling)
        original_processed = preprocess_input(original * 255.0)
        reconstructed_processed = preprocess_input(reconstructed * 255.0)
        
        # Extract features for original and reconstructed images
        original_features = self.feature_extractor(original_processed)
        reconstructed_features = self.feature_extractor(reconstructed_processed)
    
        
        # Compute perceptual loss as mean squared error between features
        perceptual_loss = 0
        for orig_feat, recon_feat in zip(original_features, reconstructed_features):
            perceptual_loss += ops.mean(ops.square(orig_feat - recon_feat))

        
        # Normalize by the number of feature layers
        perceptual_loss /= len(self.feature_layers)
        
        return perceptual_loss
    
    def compute_reconstruction_loss(self, original, reconstructed):
        
        reconstruction_loss = ops.mean(
                ops.sum(
                    keras.losses.binary_crossentropy(original, reconstructed),
                    axis=(1,2)
                )
            )
        
        return reconstruction_loss
   
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            # Encoder forward pass
            z_mean, z_log_var, z = self.encoder(data)
            
            # Decoder reconstruction
            reconstruction = self.decoder(z)
            
            # Compute KL divergence loss
            kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
            kl_loss = ops.mean(ops.sum(kl_loss, axis=1))
            kl_loss = kl_loss * self.lambda_kl
            
            # Compute perceptual loss using VGG19 features
            perceptual_loss = self.compute_perceptual_loss(data, reconstruction) * self.lambda_perceptual
            
            reconstruction_loss = self.compute_reconstruction_loss(data, reconstruction) * self.lambda_reconstruction
            
            # Total loss combines reconstruction losses and KL divergence
            total_loss = kl_loss + (perceptual_loss + reconstruction_loss)/2
        
        # Compute gradients and apply them
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.perceptual_loss_tracker.update_state(perceptual_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "perceptual_loss": self.perceptual_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result()
        }
    
    def test_step(self, data):
        # Encoder forward pass
        z_mean, z_log_var, z = self.encoder(data)
        
        # Decoder reconstruction
        reconstruction = self.decoder(z)
        
        # Compute KL divergence loss
        kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
        kl_loss = ops.mean(ops.sum(kl_loss, axis=1))
        kl_loss = kl_loss * self.lambda_kl
        
        # Compute perceptual loss using VGG19 features
        perceptual_loss = self.compute_perceptual_loss(data, reconstruction) * self.lambda_perceptual
        
        reconstruction_loss = self.compute_reconstruction_loss(data, reconstruction) * self.lambda_reconstruction
        
        # Total loss combines perceptual loss and KL divergence
        total_loss = kl_loss + (reconstruction_loss + perceptual_loss)/2
        
        # Update validation metrics
        self.val_total_loss_tracker.update_state(total_loss)
        self.val_perceptual_loss_tracker.update_state(perceptual_loss)
        self.val_kl_loss_tracker.update_state(kl_loss)
        self.val_reconstruction_loss_tracker.update_state(reconstruction_loss)
        
        return {
            "loss": self.val_total_loss_tracker.result(),
            "perceptual_loss": self.val_perceptual_loss_tracker.result(),
            "kl_loss": self.val_kl_loss_tracker.result(),
            "reconstruction_loss": self.val_reconstruction_loss_tracker.result()
        }
        
###TRAINING DATA###        
full_dataset = utils.image_dataset_from_directory(
    './Dataset/all_images',
    seed=123,
    shuffle=True,
    image_size=(128, 128),
    batch_size=None,
    label_mode=None
)

normalization_layer = Rescaling(1./255)
full_dataset = full_dataset.map(lambda x: normalization_layer(x))

# Compute counts
total_images = 200_000
train_count = 180_000
val_count = 10_000
test_count = 10_000 #approximately

# Split manually
train_data = full_dataset.take(train_count)
val_test_split = full_dataset.skip(train_count)
val_data = val_test_split.take(val_count)
test_data = val_test_split.skip(val_count)

# Batch all
train_data = train_data.batch(BATCH_SIZE, drop_remainder=True)
val_data = val_data.batch(BATCH_SIZE, drop_remainder=True)
test_data = test_data.batch(BATCH_SIZE, drop_remainder=True)


# Custom callback to plot training and validation losses
class PlotLossesCallback(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.losses = {'loss': [], 'perceptual_loss': [], 'kl_loss': [], 'reconstruction_loss': []}
        self.val_losses = {'loss': [], 'perceptual_loss': [], 'kl_loss': [], 'reconstruction_loss': []}
        self.epochs = []
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs.append(epoch)
        
        # Record training losses
        for loss_type in self.losses.keys():
            if loss_type in logs:
                self.losses[loss_type].append(logs[loss_type])
        
        # Record validation losses
        for loss_type in self.val_losses.keys():
            val_key = 'val_' + loss_type
            if val_key in logs:
                self.val_losses[loss_type].append(logs[val_key])
    
    def plot_losses(self, save_path=None):
        # Create a 2x2 grid of subplots for different loss types
        fig, axs = plt.subplots(2, 2, figsize=(20, 16))
        loss_types = list(self.losses.keys())
        
        for i, loss_type in enumerate(loss_types):
            row, col = i // 2, i % 2
            ax = axs[row, col]
            
            # Plot training loss
            if self.losses[loss_type]:
                ax.plot(self.epochs, self.losses[loss_type], label=f'Training {loss_type}')
            
            # Plot validation loss
            if self.val_losses[loss_type]:
                ax.plot(self.epochs, self.val_losses[loss_type], label=f'Validation {loss_type}')
            
            ax.set_title(f'{loss_type.replace("_", " ").title()}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, format='jpg', dpi=300)
            print(f"Loss plots saved to {save_path}")
        
        return fig


if Train == True:
    with strategy.scope():
        # Create improved models
        encoder = create_encoder(latent_dim=latent_dim)
        decoder = create_decoder(latent_dim=latent_dim)

        # Compile with better optimizer
        vae = VAE(encoder, decoder)
        vae.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4))

        vae.encoder.summary()
        vae.decoder.summary()
        vae.summary()

        load = False
        
        # Initialize plot losses callback
        plot_losses_callback = PlotLossesCallback()

        if load == True:
            #load weights
            vae.load_weights('results_encode_decode_SERVER/1/vae_weights.weights.h5')
            
        else:
            # Train with validation data
            history = vae.fit(
                train_data, 
                epochs=epochs,
                validation_data=val_data,
                callbacks=[plot_losses_callback]
            )
            
            # Plot and save the losses
            output_dir = 'results_encode_decode'
            os.makedirs(output_dir, exist_ok=True)
            result_counter_path = os.path.join(output_dir, 'result_counter.txt')

            if os.path.exists(result_counter_path):
                with open(result_counter_path, 'r') as file:
                    result_no = int(file.read().strip()) + 1
            else:
                result_no = 0

            with open(result_counter_path, 'w') as file:
                file.write(str(result_no))

            result_dir = os.path.join(output_dir, str(result_no))
            os.makedirs(result_dir, exist_ok=True)
            
            # Plot the loss curves
            plot_losses_callback.plot_losses(save_path=os.path.join(result_dir, 'loss_curves.jpg'))
            
            # Save weights in result dir
            vae.save_weights(os.path.join(result_dir, 'vae_weights.weights.h5'))
            vae.encoder.save_weights(os.path.join(result_dir, 'encoder_weights.weights.h5'))
            vae.decoder.save_weights(os.path.join(result_dir, 'decoder_weights.weights.h5'))
            
            # Generate and save sample reconstructions
            # Take a sample batch from validation data
            for val_batch in val_data.take(1):
                sample_images = val_batch[:10]  # Take 10 sample images
                
                # Encode and decode the samples
                z_mean, z_log_var, z = vae.encoder(sample_images)
                reconstructions = vae.decoder(z)
                
                # Plot original vs reconstructed images
                fig, axes = plt.subplots(2, 10, figsize=(20, 4))
                
                for i in range(10):
                    # Original images on top row
                    axes[0, i].imshow(sample_images[i])
                    axes[0, i].set_title('Original')
                    axes[0, i].axis('off')
                    
                    # Reconstructed images on bottom row
                    axes[1, i].imshow(reconstructions[i])
                    axes[1, i].set_title('Reconstructed')
                    axes[1, i].axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(result_dir, 'sample_reconstructions.jpg'), format='jpg', dpi=300)
                plt.close()
                break
            
            # Save a history log
            import json
            
            # Convert tensor values to Python native types
            def tensor_to_python(tensor_dict):
                python_dict = {}
                for key, value in tensor_dict.items():
                    if isinstance(value, list):
                        python_dict[key] = [float(v) if hasattr(v, 'numpy') else v for v in value]
                    else:
                        python_dict[key] = float(value.numpy()) if hasattr(value, 'numpy') else value
                return python_dict
            
            history_dict = tensor_to_python(history.history)
            
            with open(os.path.join(result_dir, 'training_history.json'), 'w') as f:
                json.dump(history_dict, f)
else:
    #Load weights from previous training
    encoder = create_encoder(latent_dim=latent_dim)
    decoder = create_decoder(latent_dim=latent_dim)
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4))
    vae.load_weights('results_encode_decode/5/vae_weights.weights.h5')
    vae.summary()
    
    # Evaluate on test data
    test_results = vae.evaluate(test_data)
    print(f"Test results: {test_results}")
    
    # Generate reconstructions from test data
    for test_batch in test_data.take(1):
        sample_images = test_batch[:10]  # Take 10 sample images
        
        # Encode and decode the samples
        z_mean, z_log_var, z = vae.encoder(sample_images)
        reconstructions = vae.decoder(z)
        
        # Plot original vs reconstructed images
        fig, axes = plt.subplots(2, 10, figsize=(20, 4))
        
        for i in range(10):
            # Original images on top row
            axes[0, i].imshow(sample_images[i])
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')
            
            # Reconstructed images on bottom row
            axes[1, i].imshow(reconstructions[i])
            axes[1, i].set_title('Reconstructed')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        
        # Save the visualization
        output_dir = 'results_encode_decode_evaluation'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'test_reconstructions.jpg'), format='jpg', dpi=300)
        plt.show()