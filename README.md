# Controllable-Brain-Tumor-Synthesis
Theory Project of ECE 60131: Inference and Learning in Generative Models

The main motivation here was to build a pipeline where an experienced annotator can provide a realistic 2D mask with a brain tumor, and the model will generate a corresponding realistic 2D brain MRI slice.

I played around with three different methods to do this.

* VAE + Diffusion: Trained a diffusion model in the latent space to make the model optimized and fast.
* VAE + GAN: Trained a GAN model in the latent space for the same reason. In both cases, the VAE architecture was the same.
* GAN: Directly generate 2D MRI slice for a 2D mask input.

If the reconstructed image is good enough, a segmentation model should be able to segment the tumor close to the given mask. Keeping that in mind, A U-net segmentation model was pre-trained on the dataset (Medical Segmentation Decathlon – Task 01 (Brain Tumor) derived from the BraTS (Brain Tumor Segmentation) Challenge data), and then used to segment the reconstructed image. I used HD95 and NSD as evaluation metrics.

The presentation slide and the related codes are uploaded here with some random output for each method.

Project Idea:

I wanted to build a controllable generative pipeline for 2D brain MRI synthesis that can generate realistic images with user-specified tumor locations, and evaluate the quality of these images using clinically meaningful metrics (HD95, NSD)—not generic ones like FID.

Methodology:

I compare three modern generative modeling approaches, all built on real data from the Medical Segmentation Decathlon (Task 01):

* Direct Conditional GAN (Mask → Image)
A pix2pix-style GAN that maps a tumor mask directly to a 256×256 MRI slice.

* Latent-Space GAN (VAE + GAN)
A GAN that operates in the latent space of a frozen VAE.
Mask is encoded by VAE → generator produces latent image → VAE decoder renders final image.

* Latent Diffusion Model (VAE + Diffusion)
A diffusion model trained to denoise VAE latents conditioned on downsampled tumor masks.

All models are controllable: the input is a binary tumor mask, and the output is a synthetic MRI with a tumor in that location.

For evaluation, I train a U-Net segmenter on real data, feed synthetic images to the segmenter, and compute HD95 (worst-case boundary error) and NSD (1mm surface overlap) between the predicted mask and the input conditioning mask. This ensures evaluation is clinically grounded, not just visually appealing.

Dependency:
* Python ≥ 3.10
* CUDA ≥ 12.1

Dataset:
Medical Segmentation Decathlon – Task 01 (Brain Tumor)

Required Scripts:
* train_direct_vae.py → Deterministic VAE
* train_gan.py → Direct Conditional GAN
* train_vae_gan.py → Latent-Space GAN
* train_diffusion.py → Latent Diffusion Model
* train_segmenter.py → U-Net Segmenter (for evaluation)
* evaluate_synthetic.py → HD95/NSD evaluator
