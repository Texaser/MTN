# MTN (Multi-Scale Triplane Network)
This repository contains the official implementation of Progressive Text-to-3D Generation for Automatic 3D Prototyping (https://arxiv.org/abs/2309.14600).
### [Paper](https://arxiv.org/abs/2309.14600)

### Video results


https://github.com/Texaser/MTN/assets/50570271/bdc776a6-ee2d-43ff-9ee3-21784799d3cb

https://github.com/Texaser/MTN/assets/50570271/197fa808-154b-4671-8446-8350b1e166d6



For more videos, please refer to https://www.youtube.com/watch?v=LH6-wKg30FQ

### Instructions:
0. Make sure cuda-toolkit is exported correctly:
```
nvcc -V
```
The output is like:
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Fri_Feb_21_20:23:50_PST_2025
Cuda compilation tools, release 12.8, V12.8.93
Build cuda_12.8.r12.8/compiler.35583870_0
```
1. Install the requirements:
```
conda create --name MTN python=3.9
conda activate MTN
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
conda install -c conda-forge gcc=11.2.0 gxx=11.2.0
git clone https://github.com/Texaser/MTN.git
cd MTN
pip install -r requirements.txt --no-build-isolation
```
If compilation fails, you must **override the existing torch version to make sure it match your nvcc version** (You can get the installation instructions from https://pytorch.org/get-started/previous-versions/).

To use [DeepFloyd-IF](https://github.com/deep-floyd/IF), you need to accept the usage conditions from [hugging face](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0), and login with `huggingface-cli login` in command line.

2. Start training!
```
# choose stable-diffusion version
python main.py --text "a rabbit, animated movie character, high detail 3d model" --workspace trial -O --sd_version 2.1

# use DeepFloyd-IF for guidance:

python main.py --text "a rabbit, animated movie character, high detail 3d model" --workspace trial -O --IF
python main.py --text "a rabbit, animated movie character, high detail 3d model" --workspace trial -O --IF --vram_O # requires ~24G GPU memory
python main.py -O --text "a rabbit, animated movie character, high detail 3d model" --workspace trial_perpneg_if_rabbit --iters 6000 --IF --batch_size 1 --perpneg
python main.py -O --text "a zoomed out DSLR photo of a baby bunny sitting on top of a stack of pancakes" --workspace trial_perpneg_if_bunny --iters 6000 --IF --batch_size 1 --perpneg
python main.py -O --text "A high quality photo of a toy motorcycle" --workspace trial_perpneg_if_motorcycle --iters 6000 --IF --batch_size 1 --perpneg

# larger absolute value of negative_w is used for the following command because the defult negative weight of -2 is not enough to make the diffusion model to produce the views as desired
python main.py -O --text "a DSLR photo of a tiger dressed as a doctor" --workspace trial_perpneg_if_tiger --iters 6000 --IF --batch_size 1 --perpneg --negative_w -3.0

# after the training is finished:
# test (exporting 360 degree video)
python main.py --workspace trial -O --test
# also save a mesh (with obj, mtl, and png texture)
python main.py --workspace trial -O --test --save_mesh
# test with a GUI (free view control!)
python main.py --workspace trial -O --test --gui
```
### Tested environments
* python 3.9 & torch 1.13 & CUDA 11.5 on a V100.
* python 3.9 & torch 1.13 & CUDA 11.7 on a 3090/4090. 

### Tips
The training process can sometimes be unstable due to the original code pipeline (StableDreamfusion). In such cases, you might try adjusting the lr to 3e-4 or 5e-4. Setting the lr to 1e-5 is too small for the model to converge effectively. If the model fails, consider using a different prompt or a different random seed.

### Known Issue: CUDA version mismatch when building `nvdiffrast`

In some environments, the build process of **nvdiffrast** may fail due to a strict CUDA version check when the detected system CUDA Toolkit version (e.g., CUDA 12.8) does not match the CUDA version PyTorch was compiled against (e.g., CUDA 11.7).

A temporary workaround that has been used by some developers (**not recommended as a permanent solution**) is to bypass the CUDA version check inside PyTorch.

Locate the `cpp_extension.py` file in your PyTorch installation, usually at: ~/miniconda3/envs/<your_env_name>/lib/python3.9/site-packages/torch/utils/cpp_extension.py

You can confirm the path by running:

```python
python -c "import torch; print(torch.__file__)"
```

Open the file and find the function `_check_cuda_version` (typically around lines 380–400). Locate the line that raises the error:

```python
raise RuntimeError(CUDA_MISMATCH_MESSAGE.format(cuda_str_version, torch.version.cuda))
```

Replace it with a no-op statement, for example:

```python
# raise RuntimeError(CUDA_MISMATCH_MESSAGE.format(cuda_str_version, torch.version.cuda))
print("Warning: CUDA version check temporarily bypassed (build only)")
# or simply:
# pass
```

Save the file and retry the installation:
```python
pip install -r requirements.txt --no-build-isolation
```

## Star History
If you like this code, please give a star~
<img width="749" alt="image" src="https://github.com/user-attachments/assets/edcaf836-ac45-49f9-9a64-59fbf599a28d" />

[![Star History Chart](https://api.star-history.com/svg?repos=Texaser/MTN&type=Date)](https://www.star-history.com/#Texaser/MTN&Date)

# Citation

If you find this work useful, a citation will be appreciated via:
```
@article{yi2026progressive,
  title={Progressive Text-to-3D Generation for Automatic 3D Prototyping},
  author={Yi, Han and Zheng, Zhedong and Xu, Xiangyu and Chua, Tat-seng},
  journal={ACM TOMM},
  year={2026}
}
```

## Acknowledgement
This code base is built upon the following awesome open-source projects:
[Stable DreamFusion](https://github.com/ashawkey/stable-dreamfusion),
[threestudio](https://github.com/threestudio-project/threestudio)

Thanks the authors for their remarkable job !
