# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Modified by Jaewook Yeom 02/02/2020

"""
This is the main entry point for Part 2 of this MP. You should only modify code
within this file for Part 2 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""


import numpy as numpy
import math
from collections import Counter


def naiveBayesMixture(train_set, train_labels, dev_set, bigram_lambda, unigram_smoothing_parameter, bigram_smoothing_parameter, pos_prior):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    bigram_lambda - float between 0 and 1

    unigram_smoothing_parameter - Laplace smoothing parameter for unigram model (between 0 and 1)

    bigram_smoothing_parameter - Laplace smoothing parameter for bigram model (between 0 and 1)

    pos_prior - positive prior probability (between 0 and 1)
    """
 
    # TODO: Write your code here


    # ---------  training phase ---------- #

    # Unigram model builder
    pBag = Counter()
    nBag = Counter()
    totalWords = 0
    posWords = 0
    negWords = 0
    
    for words, label in zip(train_set, train_labels):
        totalWords += 1
        if label == 1:
            for word in words:
                temp = word.lower()
                pBag.update([temp])
                posWords += 1
        elif label == 0:
            for word in words:
                temp = word.lower()
                nBag.update([temp])
                negWords += 1
    
    # Bigram model builder
    pBagBigram = Counter()
    nBagBigram = Counter()
    totalWordsBigram = 0
    posWordsBigram = 0
    negWordsBigram = 0
    
    for words, label in zip(train_set, train_labels):
        reviews = [words[i : i + 2] for i in range(len(words) - 1)]
        totalWordsBigram += 1
        if label == 1:
            for pair in reviews:
                pBagBigram.update([str(pair)])
                posWordsBigram += 1
        elif label == 0:
            for pair in reviews:
                nBagBigram.update([str(pair)])
                negWordsBigram += 1


    # ------------ dev phase ------------- #

    label_list = []

    for review in dev_set:

        # Unigram counter
        pWordTypePositive = []
        nWordTypeNegative = []
        for word_ in review:
            word = word_.lower()
            pWordTypePositive.append((pBag[word] + unigram_smoothing_parameter)/(posWords + unigram_smoothing_parameter * (len(pBag) + len(nBag))))
            nWordTypeNegative.append((nBag[word] + unigram_smoothing_parameter)/(negWords + unigram_smoothing_parameter * (len(pBag) + len(nBag))))
        productPos = 0
        for pos in pWordTypePositive:
            productPos += math.log(pos)
        productNeg = 0
        for neg in nWordTypeNegative:
            productNeg += math.log(neg)

        # Bigram counter
        pWordTypePositive_bigram = []
        nWordTypeNegative_bigram = []
        pairs = [review[i : i + 2] for i in range(len(review) - 1)]
        for word_ in pairs:
            word = str(word_)
            pWordTypePositive_bigram.append((pBagBigram[word] + bigram_smoothing_parameter)/(posWordsBigram + bigram_smoothing_parameter * (len(pBag) + len(nBag))))
            nWordTypeNegative_bigram.append((nBagBigram[word] + bigram_smoothing_parameter)/(negWordsBigram + bigram_smoothing_parameter * (len(pBag) + len(nBag))))
        productPos_bigram = 0    
        for pos in pWordTypePositive_bigram:
            productPos_bigram += math.log10(pos)
        productNeg_bigram = 0
        for neg in nWordTypeNegative_bigram:
            productNeg_bigram += math.log10(neg)
        

        # Unigram + Bigram
        productPos_ = productPos + productPos_bigram
        productNeg_ = productNeg + productNeg_bigram

        if productPos_ >= productNeg_:
            x = 1
        else:
            x = 0

        label_list.append(x)
    return label_list