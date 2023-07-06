# DreamEdit: Subject-driven Image Editing
[![arXiv](https://img.shields.io/badge/arXiv-2306.12624-b31b1b.svg)](https://arxiv.org/abs/2306.12624)

Replace the subject in a given image to a customized one or add your customized subject to any provided background!

![image](https://github.com/DreamEditBenchTeam/DreamEdit/assets/34955859/b66e3809-967d-46d5-a3ba-87879550106b)

Models, code, and dataset for [DreamEdit: Subject-driven Image Editing](https://arxiv.org/abs/2306.12624).

Check [project website](https://dreameditbenchteam.github.io/) for demos and data examples.


## Requirements
A suitable conda environment named `dream_edit` can be created and activated with:

```shell
conda env create -f environment.yaml
conda activate dream_edit

# To update env
conda env update dream_edit --file environment.yml  --prune
```
> ^There is some problem with the environment file setup currently, we will fix it soon.

> For now to get the code able to run:
> Our repo requires dependencies from different repos. Please follow the official installation of:
> * [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
> * [LangSAM](https://github.com/luca-medeiros/lang-segment-anything/tree/main)
> * [Stable Diffusion](https://github.com/CompVis/stable-diffusion)
> * [GlideGen's fork of diffuser](https://github.com/gligen/diffusers)



## Huggingface Dataset
Our dataset is on Huggingface now: https://huggingface.co/datasets/tianleliphoebe/DreamEditBench
```python
from datasets import load_dataset
dataset = load_dataset("tianleliphoebe/DreamEditBench")
```

## How to run
Go to `experiment-results-analysis` folder:
```
cd experiment-results-analysis/experiments
```

Run the script:
```
sh replace_dog8_config_01.sh
```
You can change the input path for data, model, and other parameter setting in the corresponding config file (e.g. replace_dog8_config_01.yaml)
An example fine-tuned dreambooth model checkpoint for dog8 can be downloaded at [here](https://drive.google.com/file/d/1aSyA6CsCchYC1l9DxJiy0CrJsht0K0sj/view?usp=sharing).



## BibTeX

If you find this paper or repo useful for your research, please consider citing our paper:
```
@misc{li2023dreamedit,
      title={DreamEdit: Subject-driven Image Editing}, 
      author={Tianle Li and Max Ku and Cong Wei and Wenhu Chen},
      year={2023},
      eprint={2306.12624},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
