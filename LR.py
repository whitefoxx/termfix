# -*- coding: utf-8 -*-
"""
Created on Thu Nov 03 20:47:51 2011

@author: cyb
"""

import os
import cPickle
import math

corpus_path = "E:/data/email/corpus/trec06c/trec06c/"

def gen_train():
    if os.path.isfile(corpus_path + "lr_train.pk"):
        pickle_f = open(corpus_path + "lr_train.pk", "r")
        train = cPickle.load(pickle_f)
        return train
    f = open(corpus_path + "train", "r")
    train = []
    for line in f:
        toks = line.split()
        train_f = []
        train_f.append(toks[0])
        for t in toks[1:]:
            train_f.append(int(t.split(':')[1]))
        train.append(train_f)
    pickle_f = open(corpus_path + "lr_train.pk", "w")
    cPickle.dump(train, pickle_f)
    return train

def gen_test():
    if os.path.isfile(corpus_path + "lr_test.pk"):
        pickle_f = open(corpus_path + "lr_test.pk", "r")
        test = cPickle.load(pickle_f)
        return test
    f = open(corpus_path + "test", "r")
    test = []
    for line in f:
        toks = line.split()
        test_f = []
        test_f.append(toks[0])
        for t in toks[1:]:
            test_f.append(int(t.split(':')[1]))
        test.append(test_f)
    pickle_f = open(corpus_path + "lr_test.pk", "w")
    cPickle.dump(test, pickle_f)
    return test

def logistic(x):
    return 1.0 / (1 + math.exp(-x))
    
def LR_train(train, theta, lrate, max_loop = 10):
    dimension = len(train[0]) - 1
    classify = {}
    for l in range(max_loop):
        p2n = 0; n2p = 0; p2p = 0; n2n = 0
        for t in train:
            s = sum([theta[i] * t[i+1] for i in range(dimension)])
            p = logistic(s)
            d = 0
            if t[0] == '+1' and p < 0.5:
                p2n += 1
                d = 1 - p
            elif t[0] == '-1' and p >= 0.5:
                n2p += 1
                d = -p
            elif t[0] == '+1':
                p2p += 1
            else:
                n2n += 1
            for i in range(dimension):
                theta[i] += lrate * d * t[i+1]
        classify['theta'] = theta
        classify['p+1'] = float(p2p) / (p2p + n2p)
        classify['p-1'] = float(n2n) / (n2n + p2n)
        classify['r+1'] = float(p2p) / (p2p + p2n)
        classify['r-1'] = float(n2n) / (n2n + n2p)
        classify['f+1'] = 2 * classify['p+1'] * classify['r+1'] / (classify['p+1'] + classify['r+1'])
        classify['f-1'] = 2 * classify['p-1'] * classify['r-1'] / (classify['p-1'] + classify['r-1'])
        print str(n2p) + '/' + str(p2p + n2p), str(p2n) + '/' + str(n2n + p2n), len(train)
        print str(classify['p+1']),str(classify['r+1']),str(classify['f+1']),str(classify['p-1']),\
            str(classify['r-1']),str(classify['f-1'])
    return classify

def LR_test(test, theta):
    dimension = len(test[0]) - 1
    p2n = 0; n2p = 0; p2p = 0; n2n = 0
    for t in test:
        s = sum([theta[i] * t[i+1] for i in range(dimension)])
        p = logistic(s)
        if p > 0.5:
            p_label = '+1'
        else:
            p_label = '-1'
        if t[0] == p_label:
            if t[0] == '+1':
                p2p += 1
            else:
                n2n += 1
        else:
            if t[0] == '+1':
                p2n += 1
            else:
                n2p += 1
    print str(n2p) + '/' + str(p2p + n2p), str(p2n) + '/' + str(n2n + p2n), len(test)
    
if __name__ == '__main__':
    train = gen_train()
    test = gen_test()
#    cls1 = winnow_train(train, 2, 0, 7)
    theta = [0.0 for i in range(len(train) - 1)]
    cls = LR_train(train, theta, 0.05, 15)
    LR_test(test, cls['theta'])
    