## Medical Context Natural Language Inference
***

Natural Language Processing [INM434]

#### Overview

This project explores the use of pre-trained transformer-based models for natural language inference on medical annotations. The contextual inference task is framed as a three-class problem consisting of neutral, entailment, and contradiction labels. Using the bi-directional transformer architecture from the BERT family of models, this research features variants such as *distilBERT* and *tinyBERT* which are lightweight, parameter-efficient implementations derived through knowledge distillation. Given the clinical context of this research, we utilize a pre-trained *bioClinicalBERT* tokenizer to instill domain-specific language representations and improve semantic understanding of the medical corpus.

<br>

Models were trained / fine-tuned locally on native Apple Silicon hardware with GPU acceleration via Torch Metal Performance Shaders (mps) with paramaters:

- `batch size: 24`
- `learning rate: 5e^-5` 
- `epochs: 20`
- `optimizer: AdamW`
- `weight decay: 0.01`

<br>

The data used in this study is [MedNLI](https://jgc128.github.io/mednli/) — a medical text dataset for identifying textual entailment. The corpus maintains a similar structure to generalized NL inference datasets such as [SNLI](https://nlp.stanford.edu/projects/snli/) and [Multi-NLI](https://archive.nyu.edu/handle/2451/41736) but adapted specifically for medical contexts. The original pre-print of MedNLI can be found at [ArXiv CS](https://arxiv.org/abs/1808.06752).



Experimental results demonstrated that the fine-tuned `tinyBERT` architecture with approximatelt 4.4m parameters achieved performances comparable to preliminary larger language models such as GPT-3.5, obtaining a weighted F-measure of approximately 0.69 on the MedNLI validation benchmark, while requiring substantially fewer computational resources.

<br>
  
***

#### Directory Structure

```zsh
> mnli
    ├── core
    │   ├── base.py
    │   ├── mednli.py
    │   └── run.py
    ├── data
    │   └── test.csv
    ├── models
    │   ├── bert
    │   │   ├── config.json
    │   │   └── model.safetensors
    │   ├── distilbert
    │   │   └── config.json
    │   └── pickle
    │       └── bert.pkl
    ├── README.md
    ├── paper.pdf
    └── requirements.txt
```



<br>

#### Frameworks

This project was built upon core ml libraries from huggingface, namely:

- [hf-transformers](https://huggingface.co/docs/transformers/index)
- [hf-tokenizers](https://huggingface.co/docs/tokenizers/index)
- [hf-datasets](https://huggingface.co/docs/datasets/index)

with training and optimizations done through
- [safetensors](https://huggingface.co/docs/safetensors/index)
- [automodel](https://huggingface.co/docs/transformers/model_doc/auto)
- pytorch
- sklearn vectorization

<br>

#### Build
This codebase was developed using python version `3.12.4`

to build environment and run, use:

```
git clone <repo-url>
cd mnli

uv venv --python 3.12.4
source .venv/bin/activate

uv pip install -r requirements.txt

uv run core/run.py

```

<br>

***

#### Acknowledgements

This research was conducted with the support of City, University of London as part of the INM434 Natural Language Processing module under the Dept. of Computer Science. 

