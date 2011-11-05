# -*- coding: utf-8 -*-
"""
Created on Sat Nov 05 15:09:06 2011

@author: cyb
"""

import os
import cPickle

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

def pam_train(train, afa, bta, max_loop=10):
    print 'PAM train begin ...',"afa=" + str(afa),"bta=" + str(bta),"max_loop=" + str(max_loop)
    dimension = len(train[0]) - 1
    weight = [0 for i in range(dimension)]
    classify = {}
    for l in range(max_loop):
        p2n = 0; n2p = 0; p2p = 0; n2n = 0
        for t in train:
            label = int(t[0])
            s = sum([weight[i] * t[i+1] for i in range(dimension)]) * label
            if s < afa:
                for i in range(dimension):
                    weight[i] += bta * label * t[i+1]
                if label == 1:
                    p2n += 1
                else:
                    n2p += 1
            elif label == 1:
                p2p += 1
            else:
                n2n += 1

        classify['w'] = weight
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

def pam_predict(test, weight, afa):
    dimension = len(test[0]) - 1
    p2n = 0; n2p = 0; p2p = 0; n2n = 0
    for t in test:
        label = int(t[0])
        s = sum([weight[i] * t[i+1] for i in range(dimension)]) * label
        if s < afa:
            if label == 1:
                p2n += 1
            else:
                n2p += 1
        elif label == 1:
            p2p += 1
        else:
            n2n += 1
    print str(n2p) + '/' + str(p2p + n2p), str(p2n) + '/' + str(n2n + p2n), len(test)

if __name__ == '__main__':
    train = gen_train()
    test = gen_test()
    cls = pam_train(train, 1.5, 0.1, 15)
    print "single model result"
    pam_predict(test, cls['w'], 1.5)
    
    