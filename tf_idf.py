# tf_idf_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Modified by Jaewook Yeom 02/02/2020

"""
This is the main entry point for the Extra Credit Part of this MP. You should only modify code
within this file for the Extra Credit Part -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np
import math
from collections import Counter
import time



def compute_tf_idf(train_set, train_labels, dev_set):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    Return: A list containing words with the highest tf-idf value from the dev_set documents
            Returned list should have same size as dev_set (one word from each dev_set document)
    """



    # TODO: Write your code here
    
    max_tfidf_list = []

    bag = Counter()
    totalReviews = 0

    for review in train_set:
        totalReviews += 1
        temp = set(review)
        for word in temp:
            bag.update([word])
        
    for review in dev_set:
        wordBag = Counter()
        wordCount = 0
        tfidf_list = []
        word_list = []

        for word in review:
            wordCount += 1
            wordBag.update([word])
        
        for word in review:
            tfidf = (wordBag[word] / wordCount) * math.log(totalReviews / (1 + bag[word]))
            tfidf_list.append(tfidf)
            word_list.append(word)

        max = 0
        for i in range(len(tfidf_list)):
            if tfidf_list[i] >= max:
                max = tfidf_list[i]
                mintfidf_word = word_list[i]

        max_tfidf_list.append(mintfidf_word)

    return max_tfidf_list