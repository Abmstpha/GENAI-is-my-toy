# GENAI-is-my-toy 🤖
*Basically a repo for my gems with Gen AI—tweaking it, running experiments of all types, and breaking things to see how they work.*

## 🧪 Experiments

### [Lab 1: GAN vs VAE Foundations](gan%20vs%20vae/)
A comparative study of **Generative Adversarial Networks (GANs)** and **Variational Autoencoders (VAEs)** on image generation tasks.

**📂 Key Files:**
- 📓 **[Notebook](gan%20vs%20vae/Abdellahi%20lab1_gan_vae.ipynb)**: Complete implementation of GAN and VAE training loops, including hyperparameter tuning for Latent Dimension ($Z$).
- 📄 **[Final Report](gan%20vs%20vae/Abdellahi-LAB1-report.pdf)**: Comprehensive PDF report analyzing stability, mode collapse, and reconstruction quality.

**✨ Highlights:**
- **Datasets**: MNIST (Digits) & Fashion-MNIST (Clothing).
- **Analysis**:
  - **Latent Interpolation**: Visualizing smooth transitions between classes in VAEs.
  - **Stability**: Investigating mode collapse in GANs at low latent dimensions ($Z=32$).
  - **Quantitative**: Proxy FID-like score comparison between models.
