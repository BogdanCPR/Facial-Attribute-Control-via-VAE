# Variational Autoencoder for Facial Attribute Manipulation

This repository contains the code and model implementation for a deep learning system based on a Variational Autoencoder (VAE), developed for **controlled manipulation of facial attributes** in images. The model is trained on the CelebA dataset and enables operations such as adding/removing smiles, changing hair color, or applying makeup, directly in the latent space.

## ✨ Key Features

- Custom VAE architecture with convolutional encoder and decoder
- Composite loss function:
  - Binary Cross-Entropy for reconstruction
  - KL divergence for latent regularization
  - Perceptual loss using VGG19 for semantic fidelity
- Attribute vector extraction from the latent space
- Attribute orthogonalization based on Pearson correlation
- Controlled image editing via vector operations in latent space
- Visualization using t-SNE and UMAP for semantic structure analysis

## 🧠 Model Overview

- Input images are aligned and resized to **128x128 RGB**
- Encoder outputs `z_mean` and `z_log_var` vectors for sampling
- Decoder reconstructs the input from the sampled latent vector
- Attributes are represented as directional vectors computed by averaging subsets of the latent codes
- Orthogonalization reduces interference between correlated attributes

## 📊 Results

- High-quality reconstruction and attribute manipulation
- Smooth and realistic interpolations in latent space
- Evaluation of three model variants with different KL/reconstruction trade-offs
- Improved editing consistency after attribute orthogonalization

## 🔧 Requirements

- Python 3.8+
- TensorFlow / Keras
- NumPy, Matplotlib, Scikit-learn
- tqdm
- OpenCV (for preprocessing and alignment)


## 📁 Dataset

This project uses the [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). Make sure to download the aligned images and attribute labels.


## 📝 Citation

If you use this work in your research, please cite the corresponding thesis or acknowledge this repository.

## 📚 Related Work

- [AttGAN: Facial Attribute Editing by Only Changing What You Want](https://doi.org/10.1109/TIP.2019.2916751)
- [PA-GAN: Progressive Attention GAN for Facial Attribute Editing](https://arxiv.org/abs/2007.05892)
- [MU-GAN: Facial Attribute Editing with Multi-Attention](https://arxiv.org/pdf/2009.04177)
- [CAFE-GAN: Arbitrary Face Attribute Editing](https://arxiv.org/pdf/2011.11900)

## 📬 Contact

For questions or suggestions, feel free to open an issue or contact the author.
