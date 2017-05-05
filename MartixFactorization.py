#-*- coding: utf-8 -*-
import sys, random, math
import numpy as np
from operator import itemgetter

class Dataset():
    def __init__(self):
        self.dataset = {}
        self.trainset = {}
        self.testset = {}
        self.nUsers = 0
        self.nItems = 0
        self.nTrainSet = 0
        self.nTestSet = 0
    
    def loadfile(self, filename):
        # load a file, return a generator
        fp = open(filename, 'r')
        for i,line in enumerate(fp):
            yield line.strip('\r\n')
            if i%100000 == 0:
                print >> sys.stderr, 'loading %s(%s)' % (filename, i)
        fp.close()
        print >> sys.stderr, 'load %s succ' % filename

    def generate_dataset(self, filename, pivot=0.7):
        print >> sys.stderr, 'spliting training set and test set'
        
        itemSet = set()
        for line in self.loadfile(filename):
            user, item, rating, timestamp = line.split('::')
            
            self.dataset.setdefault(int(user), {})
            self.dataset[int(user)][int(item)] = long(long(timestamp) * 10 + int(float(rating)))
        
        # split the data by pivot
        for user, items in self.dataset.items():
            for item, timestamp in sorted(items.items(), key=itemgetter(1), reverse=True)[0:int(len(items) * (1 - pivot))]:
                self.testset.setdefault(user, {})
                self.testset[user][item] = int(timestamp % 10)
                itemSet.add(item)
                self.nTestSet = self.nTestSet + 1
            for item, timestamp in sorted(items.items(), key=itemgetter(1), reverse=True)[int(len(items) * (1 - pivot)):]:
                self.trainset.setdefault(user, {})
                self.trainset[user][item] = int(timestamp % 10)
                itemSet.add(item)
                self.nTrainSet = self.nTrainSet + 1
                
        self.nItems = len(itemSet)
        if len(self.trainset) != len(self.testset):
            ColorPrint('Dataset Error')
            exit()
        else:
            self.nUsers = len(self.trainset)

        print >> sys.stderr, 'train set = %s, users = %s' % (self.nTrainSet, len(self.trainset))
        print >> sys.stderr, 'test set = %s, users = %s' % (self.nTestSet, len(self.testset))
        
class MatrixFactoriztion:
    def __init__(self, data):
        self.nUsers = data.nUsers
        self.nItems = data.nItems
        
        self.trainSet = np.zeros((self.nUsers, self.nItems), dtype=np.int8)
        self.testSet = np.zeros((self.nUsers, self.nItems), dtype=np.int8)
        
        for user, items in data.trainset.items():
            for item, rating in items.items():
                self.trainSet[user, item] = rating
                
        for user, items in data.testset.items():
            for item, rating in items.items():
                self.testSet[user, item] = rating
    
    def factorization(self):
        
