# MTN (Multi-Scale Triplane Network)
This repository contains the official implementation of Progressive Text-to-3D Generation for Automatic 3D Prototyping.
### [Paper](https://arxiv.org/abs/2309.14600)

### Video results


https://github.com/Texaser/MTN/assets/50570271/bdc776a6-ee2d-43ff-9ee3-21784799d3cb

https://github.com/Texaser/MTN/assets/50570271/197fa808-154b-4671-8446-8350b1e166d6



For more videos, please refer to https://www.youtube.com/watch?v=LH6-wKg30FQ

### Instructions:
1. Install the requirements:
```
pip install -r requirements.txt
```
To use [DeepFloyd-IF](https://github.com/deep-floyd/IF), you need to accept the usage conditions from [hugging face](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0), and login with `huggingface-cli login` in command line.

2. Start training!
```
# choose stable-diffusion version
python main.py --text "a hamburger" --workspace trial -O --sd_version 2.1

# use DeepFloyd-IF for guidance:

python main.py --text "a hamburger" --workspace trial -O --IF
python main.py --text "a hamburger" --workspace trial -O --IF --vram_O # requires ~24G GPU memory
python main.py -O --text "a tiger cub" --workspace trial_perpneg_if_tiger --iters 6000 --IF --batch_size 1 --perpneg
python main.py -O --text "a shiba dog wearing sunglasses" --workspace trial_perpneg_if_shiba --iters 6000 --IF --batch_size 1 --perpneg
python main.py -O --text "a octopus toy" --workspace trial_perpneg_if_octopus --iters 6000 --IF --batch_size 1 --perpneg

# larger absolute value of negative_w is used for the following command because the defult negative weight of -2 is not enough to make the diffusion model to produce the views as desired
python main.py -O --text "a shiba dog wearing sunglasses" --workspace trial_perpneg_if_shiba --iters 6000 --IF --batch_size 1 --perpneg --negative_w -3.0

# after the training is finished:
# test (exporting 360 degree video)
python main.py --workspace trial -O --test
# also save a mesh (with obj, mtl, and png texture)
python main.py --workspace trial -O --test --save_mesh
# test with a GUI (free view control!)
python main.py --workspace trial -O --test --gui
```
### Tested environments
* torch 1.13 & CUDA 11.5 on a V100.

# Citation

If you find this work useful, a citation will be appreciated via:
```
@article{yi2023progressive,
  title={Progressive Text-to-3D Generation for Automatic 3D Prototyping},
  author={Yi, Han and Zheng, Zhedong and Xu, Xiangyu and Chua, Tat-seng},
  journal={arXiv preprint arXiv:2309.14600},
  year={2023}
}
```

## Acknowledgement
This code base is built upon the following awesome open-source projects:
[Stable DreamFusion](https://github.com/ashawkey/stable-dreamfusion),
[threestudio](https://github.com/threestudio-project/threestudio)

Thanks the authors for their remarkable job !
