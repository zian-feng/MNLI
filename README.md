## Medical Context Natural Language Inference
***

#### Overview

This project explores the use of pre-trained transformer-based models for natural language inference on medical annotations. 

The data used in this study is MedNLI -- a medical text dataset for identifying textual entailment.

This is similar to generalized datasets like SNLI and Multi-NLI but adapted specifically for medical contexts.

#### Project Structure
```zsh
├── README.md
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
├── paper.pdf
└── requirements.txt
```

#### Frameworks

This project was built upon huggingface core ml libraries:

- huggingface-transformers
- huggingface-tokenizers
- huggingface-data

with training and optimizations done using safetensors and automodel

- pytorch
- sklearn


#### Build
this codebase was developed using python version `3.12.4`

to build environment and run, use:

```

```

#### Acknowledgements

This research was conducted with the support of City, University of London as part of the INM434 Natural Language Processing module under the Dept. of Computer Science 

