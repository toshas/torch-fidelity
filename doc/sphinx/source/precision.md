# Numerical Precision of torch-fidelity

Evaluation of generative models such as GANs is an important part of the deep learning research. 
In the domain of 2D image generation, three approaches became widely spread: 
**Inception Score** ([ISC](https://arxiv.org/pdf/1606.03498.pdf)), 
**Fréchet Inception Distance** ([FID](https://arxiv.org/pdf/1706.08500.pdf)), and 
**Kernel Inception Distance** ([KID](https://arxiv.org/pdf/1801.01401.pdf)). 

These metrics, despite having a clear mathematical and algorithmic description, were initially implemented in 
TensorFlow, and inherited a few properties of the framework itself (see Interpolation) and the code they relied upon 
(see Model). 
These design decisions were effectively baked into the evaluation protocol and became an inherent part of the 
metrics specification.
As a result, researchers wishing to compare against state of the art in generative modeling are forced to perform 
evaluation using codebases of the original metric authors. 
Reimplementations of metrics in PyTorch and other frameworks exist, but they do not provide a proper level of numerical precision, 
thus making them unsuitable for reporting results and comparing them to other methods.   

This software aims to provide epsilon-exact implementations of the said metrics in PyTorch, and thus remove
inconveniences associated with generative models evaluation and development. All steps of the evaluation pipeline are
correctly tested, with relative errors and sources of remaining non-determinism summarized below.  

## Numerical Precision Breakdown

The path from inputs to metrics contains the following steps:
1. Image interpolation (resizing) and normalization (bringing RGB values to the range acceptable by the pre-trained 
model),
2. Using the pre-trained model to extract features from properly sized and normalized images,
3. Compute metrics values from the extracted features.

Let's consider how close each of these steps can be implemented in a different framework:

### Interpolation

Bilinear interpolation is a known source of problems with determinism and reproducibility, as there are quite a few ways
to align corners of an image, [each giving a different result](https://machinethink.net/blog/coreml-upsampling/). 
The effect of the issue is amplified when going from small to large image sizes, as is the case with evaluating CIFAR10: 
32x32 is upsampled to 299x299. torch-fidelity provides a custom implementation of bilinear resize function, 
implemented exactly as in TensorFlow. The tests show that on CPU this function produces bit-exact results consistent 
with TF implementation. However, on GPU the output is epsilon-exact, presumably due to fused-multiply-add floating-point 
optimizations employed in gridpoint coordinates computation on the GPU hardware side.

### Pre-trained Model

All three metrics were introduced to evaluate 2D images with Inception V3 as a feature extractor. The architecture
of Inception V3 used in TensorFlow is slightly different from both the paper and what is implemented in 
torchvision. The author of [pytorch-fid](https://github.com/mseitzer/pytorch-fid) did the heavy lifting of porting the 
original architecture to PyTorch, as well as exporting all learnable parameters into a PyTorch checkpoint. 
torch-fidelity attempts to download the checkpoint from github. However, it is also possible to perform model 
conversion locally and load the checkpoint from a file (see `util_convert_inception_weights.py` and 
`--feature-extractor-weights-path` command line key). 

As indicated by the tests (`test_convolution.py`, `test_inception.py`), the output of the pre-trained TensorFlow model
varies from time to time given the same inputs, which can be fixed using NVIDIA's `tensorflow-determinism` package. 
PyTorch uses CuDNN as a backend, and its outputs do not vary even when determinism is disabled. Slight differences
between outputs of TF and PT convolutions exist even with maximum determinism, which does not allow us to claim
bit-exactness. Further troubleshooting reasons for inexact outputs of convolution is complicated by varying behavior
between CPU and GPU, and also between different CPUs.

### Computing metrics from features
This step is usually performed on CPU using an established math library. It is neither the bottleneck of computations,
nor the source of numerical precision loss or non-determinism; torch-fidelity uses the original features-to-metric code for this 
step in most cases.

### Relative Errors Summary

The tests are performed using fixed pseudo-random subsets of 5000 CIFAR10 images (see `tests/tf1/numerical_precision`). Below is a 
list of pixel-, feature-, or metric-wise relative errors of different steps of the computational pipeline, between 
reference implementations and torch-fidelity:
- Interpolation: 1e-5
- Convolution: 1e-7
- InceptionV3 RGB 8bpp to pool3 features: 1e-4
- Inception Score: 1e-3
- Fréchet Inception Distance: 1e-6
- Kernel Inception Distance: 1e-6

The badge 'CircleCI: passing' means that the declared tolerances were met during the last test run, involving all the 
latest versions of the dependencies. 

## A note on PPL

The package also provides **Perceptual Path Length** ([PPL](https://arxiv.org/pdf/1812.04948.pdf)) metric, which was 
originally proposed by StyleGAN authors and implemented in TensorFlow. However, the authors re-implemented it later in 
PyTorch, which served as a reference implementation for the one included in torch-fidelity. Due to the complex 
integration of the original code in the StyleGAN repository, and incompatible license, black-box testing of this metric 
values was not included into the torch-fidelity test suite. Instead, one may check the following fork 
https://github.com/toshas/torch-fidelity-stylegan2-ada and run the evaluation script with torch-fidelity and the 
original evaluation code side-by-side.

Below we report the results of evaluating PPL with a variety of epsilon and type coercion flag values. The latter
denotes the choice of data type which the generated fakes are cast to before feeding into the similarity measure 
(LPIPS). All runs were performed with the `z_end` configuration of the metric on the original pre-trained CIFAR-10 
generator.

| ppl_eps | uint8     | float32  |
| ------- |:---------:| --------:|
| 1e-4    | 1146.89   |    31.04 |
| 1e-3    | 109.6     |    30.69 |
| 1e-2    | 28.92     |    27.46 |
| 1e-1    | 11.34     |    11.33 |

## Reference Source Code

Each source code file implementing a metric part of torch-fidelity is annotated with a link to the source code of the 
reference implementation of the respective metric.