## TensorFlow and PyTorch on Apple Metal Hardware Accelerated Graphics

In order to get native support for Mac M1 (or other Apple Silicon chips), you have to follow [this](https://developer.apple.com/metal/tensorflow-plugin/) guide.


```bash
brew install mambaforge
```

And then later install `mamba` via

```bash
conda install mamba -n base -c conda-forge
```

Now, you can create a new env via

```bash
mamba env create -n machine_learning -f docs/config/tf-metal-arm64.yaml
```




### Installation Commands

> If you have `Anaconda` installed already, you have to uninstall it first (at the time of writing, you can not install required dependencies on `Anaconda` installation for `conda`)
#### 1. Download `Miniforge` OR `Mambaforge` via `brew`

Before starting, make sure to install `conda` via `Miniforge` or `Mambaforge`. 

```bash
brew install miniforge
```

or consider installing `mambaforge` (for details, checkout [this](https://stackoverflow.com/a/72970797/6390175) answer).

```bash
brew install mambaforge
```

#### 1. Install the TensorFlow dependencies:

```bash
conda install -c apple tensorflow-deps
```

#### 1. Install base TensorFlow

```bash
python -m pip install tensorflow-macos
```

#### 1. Install tensorflow-metal plugin

```bash
python -m pip install tensorflow-metal
```

#### 1. Install tensorflow-addons package (Optional)

```bash
python -m pip install tensorflow-addons
```

#### 1. Install PyTorch package

```bash
python -m pip install torch
```

### Test or Validation:

```txt
> import torch

> torch.backends.mps.is_available()
True

> torch.backends.mps.is_built()
True

> import tensorflow as tf
> tf.test.is_gpu_available()

Metal device set to: Apple M1

systemMemory: 16.00 GB
maxCacheSize: 5.33 GB

> tf.config.list_physical_devices()
[PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU'), PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]

>>> tf.config.list_physical_devices('GPU')
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]

```

### Resources:

- https://developer.apple.com/metal/tensorflow-plugin/
- https://developer.apple.com/metal/
- https://blog.tensorflow.org/2021/06/pluggabledevice-device-plugins-for-TensorFlow.html
- https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/
- https://www.tensorflow.org/api_docs/python/tf/config/PhysicalDevice