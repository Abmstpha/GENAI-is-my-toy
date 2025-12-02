# GENAI-is-my-toy 🤖
*Basically a repo for my gems with Gen AI—tweaking it, running experiments of all types, and breaking things to see how they work.*

## 🧪 Experiments

### [GAN vs VAE Foundations](gan%20vs%20vae/)
A comparative study of **Generative Adversarial Networks (GANs)** and **Variational Autoencoders (VAEs)** on image generation tasks.

**📂 Key Files:**
- 📓 **[Notebook](gan%20vs%20vae/gan_vs_vae_experiments.ipynb)**: Complete implementation of GAN and VAE training loops, including hyperparameter tuning for Latent Dimension ($Z$).
- 📄 **[Final Report](gan%20vs%20vae/gan_vs_vae_report.pdf)**: Comprehensive PDF report analyzing stability, mode collapse, and reconstruction quality.

**✨ Highlights:**
- **Datasets**: MNIST (Digits) & Fashion-MNIST (Clothing).
- **Analysis**:
  - **Latent Interpolation**: Visualizing smooth transitions between classes in VAEs.
  - **Stability**: Investigating mode collapse in GANs at low latent dimensions ($Z=32$).
  - **Quantitative**: Proxy FID-like score comparison between models.

### [LLM Fine-tuning & LoRA](LLM-Finetuning/)
Fine-tuning a small causal Language Model (`distilgpt2`) on the Tiny Shakespeare dataset to explore style transfer and parameter efficiency.

**📂 Key Files:**
- 📓 **[Notebook](LLM-Finetuning/llm_finetune.ipynb)**: Automated pipeline for Baseline, Full Fine-tuning, and LoRA experiments.
- 📄 **[Final Report](LLM-Finetuning/Lab2_Report.pdf)**: Analysis of Perplexity, Catastrophic Forgetting, and Ablation studies.

**✨ Highlights:**
- **Techniques**: Full Fine-tuning vs. Low-Rank Adaptation (LoRA).
- **Analysis**:
  - **Catastrophic Forgetting**: Demonstrating how Full FT destroys general knowledge while LoRA retains slightly more (though still limited).
  - **Ablation Study**: Investigating the impact of LoRA Rank ($r=1, 8, 16, 64$) on perplexity and efficiency.
  - **Prompt Engineering**: Comparing Zero-shot vs Few-shot performance on Shakespearean style.
