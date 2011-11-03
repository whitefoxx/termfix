# -*- coding: utf-8 -*-
"""
Created on Thu Nov 03 12:45:11 2011

@author: cyb
"""
import os
import cPickle

corpus_path = "E:/data/email/corpus/trec06c/trec06c/"

def gen_train():
    if os.path.isfile(corpus_path + "winnow_train.pk"):
        pickle_f = open(corpus_path + "winnow_train.pk", "r")
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
    pickle_f = open(corpus_path + "winnow_train.pk", "w")
    cPickle.dump(train, pickle_f)
    return train

def gen_test():
    if os.path.isfile(corpus_path + "winnow_test.pk"):
        pickle_f = open(corpus_path + "winnow_test.pk", "r")
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
    pickle_f = open(corpus_path + "winnow_test.pk", "w")
    cPickle.dump(test, pickle_f)
    return test

def winnow_train(train, afa, bta, max_loop=10):
    print 'winnow train begin ...',"afa=" + str(afa),"bta=" + str(bta),"max_loop=" + str(max_loop)
    dimension = len(train[0]) - 1
    weight = [1 for i in range(dimension)]
    cta = dimension / 2
    classify = {}
    for l in range(max_loop):
        p2n = 0; n2p = 0; p2p = 0; n2n = 0
        for t in train:
            label = t[0]
            s = sum([weight[i] * t[i+1] for i in range(dimension)])
            if s > cta:
                p_label = '+1'
            else:
                p_label = '-1'
            if label == p_label:
                if label == '+1':
                    p2p += 1
                else:
                    n2n += 1
            else:
                if label == '+1':
                    mul = afa
                    p2n += 1
                else:
                    mul = bta
                    n2p += 1
                for i in range(dimension):
                    if t[i+1] == 1:
                        weight[i] *= mul

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

# It seems that combination is not work
def winnow_combine_predict(test, cls1, cls2):
    wght_1 = cls1['w']
    wght_2 = cls2['w']
    dimension = len(test[0]) - 1
    cta = dimension / 2
    p2n = 0; n2p = 0; p2p = 0; n2n = 0
    for t in test:
        s1 = sum([wght_1[i] * t[i+1] for i in range(dimension)])
        s2 = sum([wght_2[i] * t[i+1] for i in range(dimension)])
        if s1 > cta:
            if s2 > cta or cls1['p+1'] > cls2['p-1']:
                p_label = '+1'
            else:
                p_label = '-1'
        else:
            if s2 <= cta or cls1['p-1'] > cls2['p+1']:
                p_label = '-1'
            else:
                p_label = '+1'
        if p_label == t[0]:
            if t[0] == '+1':
                p2p += 1
            else:
                n2n += 1
        if p_label != t[0]:
            if t[0] == '+1':
                p2n += 1
            else:
                n2p += 1
    print str(n2p) + '/' + str(p2p + n2p), str(p2n) + '/' + str(n2n + p2n), len(test)

def winnow_predict(test, weight):
    dimension = len(test[0]) - 1
    cta = dimension / 2
    p2n = 0; n2p = 0; p2p = 0; n2n = 0
    for t in test:
        s = sum([weight[i] * t[i+1] for i in range(dimension)])
        if s > cta:
            if t[0] != '+1':
                n2p += 1
            else:
                p2p += 1
        else:
            if t[0] != '-1':
                p2n += 1
            else:
                n2n += 1
    print str(n2p) + '/' + str(p2p + n2p), str(p2n) + '/' + str(n2n + p2n), len(test)
    
if __name__ == '__main__':
    train = gen_train()
    test = gen_test()
#    cls1 = winnow_train(train, 2, 0, 7)
    cls2 = winnow_train(train, 1.23, 0.83, 15)
    print "single model result"
#    winnow_predict(test, cls1['w'])
    winnow_predict(test, cls2['w'])
    print "combine model result"
#    winnow_combine_predict(test, cls1, cls2)
    