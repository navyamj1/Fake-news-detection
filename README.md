
# Fake News Detection using NLP

This project uses Natural Language Processing (NLP) techniques and a PassiveAggressiveClassifier to detect whether a news article is fake or real.

## Dataset
The dataset used is the Fake and Real News Dataset available on Kaggle.

## Dataset Setup
Download the dataset form the link:  https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset?resource=download

## Features
- Text cleaning and preprocessing
- TF-IDF vectorization
- PassiveAggressiveClassifier model
- Evaluation using accuracy, confusion matrix, and classification report

## How to Run
1. Install requirements: `pip install -r requirements.txt`
2. Prepare dataset: `python prepdataset.py`
3. Run : `python FakeNewsDetection.py`


