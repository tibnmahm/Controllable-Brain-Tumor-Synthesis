# Controllable-Brain-Tumor-Synthesis
Theory Project of ECE 60131: Inference and Learning in Generative Models

The main motivation here was to build a pipeline where an experienced annotator can provide a realistic 2D mask with a brain tumor, and the model will generate a corresponding realistic 2D brain MRI slice.

I played around with three different methods to do this.

* VAE + Diffusion: Trained a diffusion model in the latent space to make the model optimized and fast.
* VAE + GAN: Trained a GAN model in the latent space for the same reason. In both cases, the VAE architecture was the same.
* GAN: Directly generate 2D MRI slice for a 2D mask input.

If the reconstructed image is good enough, a segmentation model should be able to segment the tumor close to the given mask. Keeping that in mind, A U-net segmentation model was pre-trained on the dataset (Medical Segmentation Decathlon – Task 01 (Brain Tumor) derived from the BraTS (Brain Tumor Segmentation) Challenge data), and then used to segment the reconstructed image. I used HD95 and NSD as evaluation metrics.

The presentation slide and the related codes are uploaded here with some random output for each method.
