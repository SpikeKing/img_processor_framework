## Installation

Our [Colab Notebook](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5)
has step-by-step instructions that install detectron2.
The [Dockerfile](https://github.com/facebookresearch/detectron2/blob/master/docker/Dockerfile)
also installs detectron2 with a few simple commands.

### Requirements
- Linux or macOS
- Python ≥ 3.6
- PyTorch ≥ 1.3
- [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
	You can install them together at [pytorch.org](https://pytorch.org) to make sure of this.
- OpenCV, needed by demo and visualization
- pycocotools: `pip install cython; pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'`
- GCC ≥ 4.9


### Build and Install Detectron2

After having the above dependencies, run:
```
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
pip install -e .

# or if you are on macOS
# MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ pip install -e .

# or, as an alternative to `pip install`, use
# python setup.py build develop
```
Note: you often need to rebuild detectron2 after reinstalling PyTorch.

### Common Installation Issues

Click each issue for its solutions:

<details>
<summary>
Undefined torch/aten/caffe2 symbols, or segmentation fault immediately when running the library.
</summary>

This can happen if detectron2 or torchvision is not
compiled with the version of PyTorch you're running.

If you use a pre-built torchvision, uninstall torchvision & pytorch, and reinstall them
following [pytorch.org](http://pytorch.org).
If you manually build detectron2 or torchvision, remove the files you built (`build/`, `**/*.so`)
and rebuild them.

If you cannot resolve the problem, please include the output of `gdb -ex "r" -ex "bt" -ex "quit" --args python -m detectron2.utils.collect_env`
in your issue.
</details>

<details>
<summary>
Undefined C++ symbols in `detectron2/_C*.so`.
</summary>
Usually it's because the library is compiled with a newer C++ compiler but run with an old C++ run time.
This can happen with old anaconda.

Try `conda update libgcc`. Then remove the files you built (`build/`, `**/*.so`) and rebuild them.
</details>

<details>
<summary>
"Not compiled with GPU support" or "Detectron2 CUDA Compiler: not available".
</summary>
CUDA is not found when building detectron2.
You should make sure

```
python -c 'import torch; from torch.utils.cpp_extension import CUDA_HOME; print(torch.cuda.is_available(), CUDA_HOME)'
```

print valid outputs at the time you build detectron2.
</details>

<details>
<summary>
"invalid device function" or "no kernel image is available for execution".
</summary>

Two possibilities:

* You build detectron2 with one version of CUDA but run it with a different version.

  To check whether it is the case,
  use `python -m detectron2.utils.collect_env` to find out inconsistent CUDA versions.
	In the output of this command, you should expect "Detectron2 CUDA Compiler", "CUDA_HOME", "PyTorch built with - CUDA"
	to contain cuda libraries of the same version.

	When they are inconsistent,
	you need to either install a different build of PyTorch (or build by yourself)
	to match your local CUDA installation, or install a different version of CUDA to match PyTorch.

* Detectron2 or PyTorch/torchvision is not built with the correct compute compatibility for the GPU model.

	The compute compatibility for PyTorch is available in `python -m detectron2.utils.collect_env`.

	The compute compatibility of detectron2/torchvision defaults to match the GPU found on the machine
	during building, and can be controlled by `TORCH_CUDA_ARCH_LIST` environment variable during building.

	Visit [developer.nvidia.com/cuda-gpus](https://developer.nvidia.com/cuda-gpus) to find out
	the correct compute compatibility for your device.

</details>

<details>
<summary>
Undefined CUDA symbols or cannot open libcudart.so.
</summary>

The version of NVCC you use to build detectron2 or torchvision does
not match the version of CUDA you are running with.
This often happens when using anaconda's CUDA runtime.

Use `python -m detectron2.utils.collect_env` to find out inconsistent CUDA versions.
In the output of this command, you should expect "Detectron2 CUDA Compiler", "CUDA_HOME", "PyTorch built with - CUDA"
to contain cuda libraries of the same version.

When they are inconsistent,
you need to either install a different build of PyTorch (or build by yourself)
to match your local CUDA installation, or install a different version of CUDA to match PyTorch.
</details>


<details>
<summary>
"ImportError: cannot import name '_C'".
</summary>
Please build and install detectron2 following the instructions above.
</details>
