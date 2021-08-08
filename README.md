# Naive Bayes classifier for movie reviews
Naive Bayes classifier for movie recommendations

Part of ECE 448: Artificial Intelligence at the University of Illinois at Urbana-Champaign

Link tor assignment: [MP3](https://courses.grainger.illinois.edu/ece448/sp2020/MPs/mp3/assignment3.html)

## Problem

We have a dataset consisting of positive and negative reviews. Using the training set,  we will learn a Naive Bayes classifier that will predict the right class label given an unseen review. We will use the development set to test the accuracy of your learned model. 

## Dataset

The dataset consists of 10000 positive and 3000 negative reviews, a subset of the [Stanford Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/), which was originally introduced by [this paper](https://www.aclweb.org/anthology/P11-1015). This data set has been split into 5000 development examples and 8000 training examples. The data set can be downloaded here: [zip](https://courses.grainger.illinois.edu/ece448/sp2020/MPs/mp3/data_zip.zip)  [tar](https://courses.grainger.illinois.edu/ece448/sp2020/MPs/mp3/data_tar.tar.gz).

## Background

The bag of words model in NLP is a simple unigram model which considers a text to be represented as a bag of independent words. That is, we ignore the position the words appear in, and only pay attention to their frequency in the text. Here each email consists of a group of words. Using Bayes theorem, we need to compute the probability of a review being positive given the words in the review: Thus we need to estimate the posterior probabilities:

<img src="https://render.githubusercontent.com/render/math?math=P(\text{Type=Positive|Words})=\frac{P(\text{Type=Positive)}}{P(\text{Words})} \prod_{\mbox{All words}} P\text{(Word|Type=Positive)}">
<img src="https://render.githubusercontent.com/render/math?math=P(\text{Type=Negative|Words})=\frac{P(\text{Type=Negative})}{P(\text{Words})} \prod_{\mbox{\text{All words}}} P(\text{Word|Type=Negative})">


## Unigram Model

**Before starting:**  Make sure you install the nltk package with 'pip install nltk' and/or 'pip3 install nltk', depending on the Python version. The tqdm package should also be downloaded.

-   **Training Phase:** The training set will be used to build a bag of words model using the emails. The purpose of the training set is to help you calculate P(Word|Type=Positive) and P(Word|Type=Negative)  during the testing (development) phase.
    
-   **Development Phase:** In the development phase, we will calculate the  P(Type=Positive|Words) and  P(Type=Negative|Words) for each document in the development set. We will classify each document in the development set as a positive or negative review depending on which posterior probability is of higher value. A list containing labels for each of the documents in the development set is returned.    

## Unigram and Bigram Models

For Part 2, we will implement the naive Bayes algorithm over a bigram model (as opposed to a unigram model like in Part 1). Then, we will combine the bigram model and the unigram model into a mixture model defined with parameter λ:

<img src="https://latex.codecogs.com/svg.image?(1-\lambda)(log(P(Y))&plus;\sum_{i=1}^{n}log(P(w_i|Y))&plus;\lambda(log(P(Y))&plus;\sum_{i=1}^{m}log(P(b_i|Y))" title="(1-\lambda)(log(P(Y))+\sum_{i=1}^{n}log(P(w_i|Y))+\lambda(log(P(Y))+\sum_{i=1}^{m}log(P(b_i|Y))" />
