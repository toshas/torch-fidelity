# Usage Examples with Command Line

The `fidelity` tool (run with `--help` to see the reference) is a command line interface to the `calculate_metrics` 
function.

## Inception Score of CIFAR-10 training split 

Command:

```
> fidelity --gpu 0 --isc --input1 cifar10-train
```

Output:

```
inception_score_mean: 11.23678
inception_score_std: 0.09514061
```

## Inception Score of a directory of images

We assume that the images are stored in `~/images/`:

```
> fidelity --gpu 0 --isc --input1 ~/images/
```

## Inception Score of a generative model 

We assume that the generative model is stored in `~/generator.pth`, whose input is a 128-dimensional standard normal 
random sample. This is equivalent to sampling 50000 images from the model, saving them in a temporary directory, and 
running the previous command:

```
> fidelity --gpu 0 --isc --input1 ~/generator.pth --input1-model-z-size 128 --input1-model-num-samples 50000 
```

## FID between a directory of images and CIFAR-10 training split

Fréchet Inception Distance between a directory of images and CIFAR-10 training split; we assume that the images are 
stored in `~/images/`:

```
> fidelity --gpu 0 --fid --input1 ~/images/ --input2 cifar10-train
```

## Efficient computation of ISC, FID, and KID

Efficient computation of ISC, FID, and KID with feature caching between the first and the second inputs, where 
the value of `--input1` can be either a registered input (e.g., `cifar10-train`), or a directory with samples, or a 
generative model stored in either `.pth` or `.onnx` formats:

```
> fidelity --gpu 0 --isc --fid --kid --input1 <input> --input2 cifar10-train
```

## Efficient computation of ISC and PPL

Efficient computation of ISC and PPL for `input1`, and FID and KID between a generative model stored in 
`~/generator.onnx` and CIFAR-10 training split:

```
> fidelity \
  --gpu 0 \
  --isc \
  --fid \
  --kid \
  --ppl \
  --input1 ~/generator.onnx \ 
  --input1-model-z-type normal \
  --input1-model-z-size 128 \
  --input1-model-num-samples 50000 \ 
  --input2 cifar10-train 
```

## Differences between CLI and API interfaces 

For the most part, `calculate_metrics` keyword arguments correspond to the command line keys. Exceptions are:
- Instances of `torch.util.data.Dataset` can be passed as inputs only programmatically;
- Instances of `torch.nn.GenerativeModelBase` can be passed as inputs only programmatically;
- CLI usage of registered inputs, feature extractors, _etc._ is limited to those preregistered in the registry; 
- CLI argument `--silent` sets kwarg `verbose=False`, whose default value is `True`;
- CLI argument `--no-datasets-download` sets kwarg `datasets_download=False`, whose default value is `True`;
- CLI argument `--no-samples-shuffle` sets kwarg `samples_shuffle=False`, whose default value is `True`;
- CLI argument `--no-cache` sets kwarg `cache=False`, whose default value is `True`;
