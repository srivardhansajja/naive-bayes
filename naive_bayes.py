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
This is the main entry point for Part 1 of MP3. You should only modify code
within this file for Part 1 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as numpy
import math
from collections import Counter

def naiveBayes(train_set, train_labels, dev_set, smoothing_parameter, pos_prior):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    smoothing_parameter - The smoothing parameter you provided with --laplace (1.0 by default)

    pos_prior - positive prior probability (between 0 and 1)
    """

    # TODO: Write your code here

    # ---------  training phase ---------- #

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

    # ------------  dev phase ------------- #

    label_list = []

    for review in dev_set:
        pWordTypePositive = []
        nWordTypeNegative = []

        for word_ in review:
            word = word_.lower()
            pWordTypePositive.append((pBag[word] + smoothing_parameter)/(posWords + smoothing_parameter * (len(pBag) + len(nBag))))
            nWordTypeNegative.append((nBag[word] + smoothing_parameter)/(negWords + smoothing_parameter * (len(pBag) + len(nBag))))

        productPos = 0
        for pos in pWordTypePositive:
            productPos += math.log10(pos)
        
        productNeg = 0
        for neg in nWordTypeNegative:
            productNeg += math.log10(neg)

        if productPos >= productNeg:
            x = 1
        else:
            x = 0

        label_list.append(x)
    return label_list