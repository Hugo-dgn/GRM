# Belief Propagation for Image Denoising and Segmentation

## Image Denoising

The image denoising part of the project serves as a sanity check for the algorithm implementations. You can find the pipeline and explanation in the notebook `denoise_MNIST.ipynb`.

## Image Segmentation

Segmentation is split into two parts: per-pixel segmentation and superpixel segmentation. In all notebooks, you will find a `category` variable that can be modified to segment different breeds of dogs or cats.

### Per-Pixel Segmentation

This is the naive approach: very computationally expensive but capable of producing good results. The pipeline and explanation are provided in `OxfordIIITPet_pixels.ipynb`.

### Superpixel Segmentation

To reduce computation time, superpixels are used as nodes in the graph instead of individual pixels. The pipeline and explanations are available in `OxfordIIITPet_super_pixels.ipynb`.

## Algorithm Implementation

Three algorithms are implemented:
- Tree Belief Propagation (Tree BP)
- Loopy Belief Propagation (Loopy BP)
- Tree-Reweighted Belief Propagation (TRW-BP)

These can be found in the `inference.py` file.