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
```

## Huggingface Dataset
Our dataset is on Huggingface now: https://huggingface.co/datasets/tianleliphoebe/DreamEditBench
```python
from datasets import load_dataset
dataset = load_dataset("tianleliphoebe/DreamEditBench")
```


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
