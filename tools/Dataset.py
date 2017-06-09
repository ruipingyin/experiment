import sys, random, math
from operator import itemgetter
from utils import ColorPrint

import numpy as np
import scipy.sparse as sp

class DataModel():
  def __init__(self, filename, implicit, pivot): 
    ''' pivot = -1: only one item for per user in test set. '''
    self.filename, self.implicit, self.pivot = filename, implicit, pivot
    
    self.dataset, self.trainset, self.testset = {}, [], []
    
    self.itemCount = 0
    self.userCount = 0
    self.ratingCount = 0
    
    self.generate_dataset()

  def generate_dataset(self):
    itemSet, userSet = set(), set()
    
    with open(self.filename, 'r') as fp:
      for i, line in enumerate(fp):
        user, item, rating, timestamp = line.strip('\r\n').split()
        self.dataset.setdefault(int(user), {})
        if self.implicit:
          self.dataset[int(user)][int(item)] = (long(timestamp), 1)
        else:
          self.dataset[int(user)][int(item)] = (long(timestamp), int(float(rating)))
        itemSet.add(int(item))
        userSet.add(int(user))
    
    userStartIndex = -1
    if max(userSet) == len(userSet):
      userStartIndex = 1
    elif (max(userSet) + 1) == len(userSet):
      userStartIndex = 0
    else:
      ColorPrint('Incompleted Dataset!')
      exit()
      
    itemStartIndex = -1
    if max(itemSet) == len(itemSet):
      itemStartIndex = 1
    elif (max(itemSet) + 1) == len(itemSet):
      itemStartIndex = 0
    else:
      ColorPrint('Incompleted Dataset!')
      exit()

    if self.pivot > 0:
      for user, items in self.dataset.items():
        items = sorted(items.items(), key=lambda x: x[1][0], reverse=True)
        for item, timestamp in items[0:int(len(items) * (1 - self.pivot))]:
          self.testset.append([user - userStartIndex, item - itemStartIndex, timestamp[1]])
        for item, timestamp in items[int(len(items) * (1 - self.pivot)):]:
          self.trainset.append([user - userStartIndex, item - itemStartIndex, timestamp[1]])
    else:
      for user, items in self.dataset.items():
        items = sorted(items.items(), key=lambda x: x[1][0], reverse=True)
        self.testset.append([user - userStartIndex, items[0][0] - itemStartIndex, items[0][1][1]])
        for item, timestamp in items[1:]:
          self.trainset.append([user - userStartIndex, item - itemStartIndex, timestamp[1]])
    
    self.userCount = len(userSet)
    self.itemCount = len(itemSet)
    
    # Validation
    ColorPrint('Load data succ!', 1)

  def dataMap(self):
    trainSet, testSet = {}, {}
    if self.implicit:
      for dataPoint in self.trainset:
        trainSet.setdefault(dataPoint[0], set())
        trainSet[dataPoint[0]].add([dataPoint[1]])
      for dataPoint in self.testset:
        testSet.setdefault(dataPoint[0], set())
        testSet[dataPoint[0]].add([dataPoint[1]])
    else:
      for dataPoint in self.trainset:
        trainSet.setdefault(dataPoint[0], {})
        trainSet[dataPoint[0]][dataPoint[1]] = dataPoint[2]
      for dataPoint in self.testset:
        testSet.setdefault(dataPoint[0], {})
        testSet[dataPoint[0]][dataPoint[1]] = dataPoint[2]
    return (trainSet, testSet)
  
  def dataList(self):
    return (np.array(self.trainset), np.array(self.testset), self.userCount, self.itemCount)
    
  def dataMatrix(self):
    mTrain = np.zeros((self.userCount, self.itemCount))
    for tri in self.trainset:
      mTrain[tri[0]][tri[1]] = tri[2]
    return (mTrain, np.array(self.testset), self.userCount, self.itemCount)

  def dataInfo(self):
    ColorPrint("Record Count: %d, User Count: %d, Item Count: %d" % (len(self.testset) + len(self.trainset), self.userCount, self.itemCount), 1)
    ColorPrint("Sparsity: %.6f" % ((len(self.testset) + len(self.trainset)) * 1.0 / self.userCount / self.itemCount), 1)
    
    max, min = -1, 999
    sum = 0
    for user, items in self.dataset.items():
      max = len(items) if len(items) > max else max
      min = len(items) if len(items) < min else min
      sum += len(items)
    ColorPrint("Item per user: %d, %d, %.2f" % (min, max, sum * 1.0 / self.userCount), 1)
    
    if sum != len(self.testset) + len(self.trainset): ColorPrint("Record Count Error.")
    
    dataset_i = {}
    max, min, sum = -1, 999, 0
    with open(self.filename, 'r') as fp:
      for i, line in enumerate(fp):
        user, item, _, _ = line.strip('\r\n').split()
        dataset_i.setdefault(int(item), [])
        dataset_i[int(item)].append(user)
    for item, users in dataset_i.items():
      max = len(users) if len(users) > max else max
      min = len(users) if len(users) < min else min
      sum += len(users)
    ColorPrint("User per item: %d, %d, %.2f" % (min, max, sum * 1.0 / self.itemCount), 1)
    
    dataHist = []
    for user, items in self.dataset.items():
      dataHist.append(len(items))
      
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.hist(dataHist, bins=1000) 
    plt.savefig('./M1.png')
  
  def dataPrep(self):
    keyList = []
    for user, items in self.dataset.items():
      if len(items) < 2: keyList.append(user)
    for user in keyList:
      del self.dataset[user]
    
    dataset_i = {}
    for user, items in self.dataset.items():
      for item in items:
        dataset_i.setdefault(int(item), [])
        dataset_i[int(item)].append(user)
        
    ColorPrint('Item Count: %d' % len(dataset_i))
    
    keyList = []
    for user, items in dataset_i.items():
      if len(items) < 2: keyList.append(user)
    for user in keyList:
      del dataset_i[user]
    
    dataset_u = {}
    for user, items in dataset_i.items():
      for item in items:
        dataset_u.setdefault(int(item), [])
        dataset_u[int(item)].append(user)
    
    ColorPrint('User Count: %d' % len(dataset_i))
    
    keyList = []
    for user, items in dataset_u.items():
      if len(items) < 10: keyList.append(user)
    for user in keyList:
      del dataset_u[user]
    
    userMap, itemMap = {}, {}
    for user, items in dataset_u.items():
      if user not in userMap: userMap[user] = len(userMap)
      for item in items:
        if item not in itemMap: itemMap[item] = len(itemMap)
    
      
    with open('./dataset/Ama/reviews_t20.txt', 'w') as out:
      for user, items in dataset_u.items():
        for item in items:
          out.write('%d\t%d\t1\t2222\n' % (userMap[user], itemMap[item]))
