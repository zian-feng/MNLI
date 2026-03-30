## Medical Context Natural Language Inference
***

#### Overview




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
this codebase was developed using python version 3.12.4

to build environment and run, use:



#### Acknowledgements

This research was conducted with the support of City, University of London as part of the INM434 Natural Language Processing module under the Dept. of Computer Science 

