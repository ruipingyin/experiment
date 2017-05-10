import sys, random, math
from operator import itemgetter
from utils import ColorPrint

import numpy as np
import scipy.sparse as sp

''' Only consider implicit rating because tianchi dataset has no rating '''

class DataBase():
  def __init__(self, filename, pivot = 0.7):
    self.dataset, self.trainset, self.testset = {}, {}, {}
    self.generate_dataset(filename, pivot)
    
    self.itemCount = 0
    self.userCount = 0
    self.ratingCount = 0

  def loadfile(self, filename):
    with open(filename, 'r') as fp:
      for i, line in enumerate(fp):
        yield line.strip('\r\n')
    ColorPrint('load %s succ' % filename, 1)

  def generate_dataset(self, filename, pivot):
    trainset_len = 0
    testset_len = 0
    itemSet = set()
    for line in self.loadfile(filename):
      user, item, _, timestamp = line.split()
      self.dataset.setdefault(int(user), {})
      self.dataset[int(user)][int(item)] = long(timestamp)
      itemSet.add(int(item))

    # split the data by pivot
    for user, items in self.dataset.items():
      for item, timestamp in sorted(items.items(), key=itemgetter(1), reverse=True)[0:int(len(items) * (1 - pivot))]:
        self.testset.setdefault(user, set())
        self.testset[user].add(item)
        testset_len = testset_len + 1
      for item, timestamp in sorted(items.items(), key=itemgetter(1), reverse=True)[int(len(items) * (1 - pivot)):]:
        self.trainset.setdefault(user, set())
        self.trainset[user].add(item)
        trainset_len = trainset_len + 1
    
    self.ratingCount = trainset_len + testset_len
    self.userCount = max(len(self.trainset), len(self.testset))
    self.itemCount = len(itemSet)
    
    ColorPrint('train set = %s, users = %s' % (trainset_len, len(self.trainset)), 1)
    ColorPrint('test set = %s, users = %s' % (testset_len, len(self.testset)), 1)
  
  def get_dataset():
    return (self.trainset, self.testset)
    
  def get_item_count():
    return itemCount
    
  def get_user_count():
    return userCount
    
  def get_rating_count():
    return ratingCount

class DataSet():
  def __init__(self, filename, pivot = 0.7):
    self.dataset, self.trainset, self.testset = {}, [], []
    
    self.itemCount = 0
    self.userCount = 0
    self.ratingCount = 0
    
    self.generate_dataset(filename, pivot)

  def loadfile(self, filename):
    with open(filename, 'r') as fp:
      for i, line in enumerate(fp):
        yield line.strip('\r\n')

  def generate_dataset(self, filename, pivot):
    itemSet, userSet = set(), set()
    
    for line in self.loadfile(filename):
      user, item, rating, timestamp = line.split()
      self.dataset.setdefault(int(user), {})
      self.dataset[int(user)][int(item)] = (long(timestamp), int(float(rating)))
      itemSet.add(int(item))
      userSet.add(int(user))

    # split the data by pivot
    for user, items in self.dataset.items():
      items = sorted(items.items(), key=lambda x: x[1][0], reverse=True)
      for item, timestamp in items[0:int(len(items) * (1 - pivot))]:
        self.testset.append([user - 1, item - 1, timestamp[1]])
      for item, timestamp in items[int(len(items) * (1 - pivot)):]:
        self.trainset.append([user - 1, item - 1, timestamp[1]])
    
    self.userCount = len(userSet)
    self.itemCount = len(itemSet)
    
    # Validation
    ColorPrint('Max User: %d, User Count: %d' % (max(userSet) , len(userSet)), 1)
    ColorPrint('Max Item: %d, Item Count: %d' % (max(itemSet) , len(itemSet)), 1)
    ColorPrint('Load data succ!', 1)
  
  def get_dataset(self):
    return (np.array(self.trainset), np.array(self.testset), self.userCount, self.itemCount)
    
if __name__ == '__main__':
  ratingfile = '../dataset/reviews.txt'

  db = DataBase()
  db.generate_dataset(ratingfile)

  