### Features_extraction
This repository contains scripts to obtains three type of summaries. 
This was develop to obtain summaries of body in Fake news challenge dataset. 
In addition, you can use this repository to calculate different similarity features between two text.

# Summaries
Text rank, BERT summary and BART summary.

# Features
cosine_similarity, jaccard_distance, hellinger_distance,
            kullback_leibler_distance, overlap, bert_cosine_similarity, soft_cosine_similarity, polarityClaim_nltk_neg, polarityClaim_nltk_pos,
polarityClaim_nltk_neu, polarityClaim_nltk_compoud, polarityBody_nltk_neg, polarityBody_nltk_pos, polarityBody_nltk_neu, polarityBody_nltk_compoud



### Requirements
* Python >= 3.8
* transformers >=4.23.1


#### Manual installation

Create a Python Environment and activate it:
```bash 
    virtualenv features--python=python3
    cd ./features
    source bin/activate
```
Install the required dependencies. 
You need to have at least version 21.0.1 of pip installed. Next you may install requirements.txt.

```bash
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

Download the FNC-dataset and the generatic summaries from this link:
```bash
wget -O data.zip "https://drive.google.com/uc?export=download&id=12q0GEXtkrMM4NHOiR4_jBFOyJTXJyp59"
unzip data.zip
rm data.zip
mv FNC_body data
```

### Contacts:
If you have any questions please contact the authors.   
  * Robiert Sepúlveda Torres rsepulveda911112@gmail.com 

### Citation:
```bash
@article{SEPULVEDATORRES2021100660,
title = {HeadlineStanceChecker: Exploiting summarization to detect headline disinformation},
journal = {Journal of Web Semantics},
volume = {71},
pages = {100660},
year = {2021},
issn = {1570-8268},
doi = {https://doi.org/10.1016/j.websem.2021.100660},
url = {https://www.sciencedirect.com/science/article/pii/S1570826821000354},
author = {Robiert Sepúlveda-Torres and Marta Vicente and Estela Saquete and Elena Lloret and Manuel Palomar},
keywords = {Natural Language Processing, Fake news, Misleading headlines, Stance detection, Applied computing, Document management and text processing, Semantic summarization},
}

@inproceedings{sepulveda2021exploring,
  title={Exploring summarization to enhance headline stance detection},
  author={Sep{\'u}lveda-Torres, Robiert and Vicente, Marta and Saquete, Estela and Lloret, Elena and Palomar, Manuel},
  booktitle={International Conference on Applications of Natural Language to Information Systems},
  pages={243--254},
  year={2021},
  organization={Springer}
}
```
  
### License:
  * Apache License Version 2.0 