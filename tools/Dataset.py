import sys, random, math
from operator import itemgetter

''' Only consider implicit rating because tianchi dataset has no rating '''

class DataBase():
  def __init__(self, filename, pivot = 0.8):
    self.dataset, self.trainset, self.testset = {}, {}, {}
    self.generate_dataset(filename, pivot)

  def loadfile(self, filename):
    with open(filename, 'r') as fp:
      for i, line in enumerate(fp):
        yield line.strip('\r\n')
        if i%100000 == 0: sys.stdout.write('loading %s(%s)\r' % (filename, i))
    print >> sys.stderr, 'load %s succ' % filename

  def generate_dataset(self, filename, pivot):
    print >> sys.stderr, 'spliting training set and test set'
    
    trainset_len = 0
    testset_len = 0
    for line in self.loadfile(filename):
      user, item, timestamp = line.split()
      
      self.dataset.setdefault(int(user), {})
      self.dataset[int(user)][int(item)] = long(timestamp)

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

    print >> sys.stderr, 'train set = %s, users = %s' % (trainset_len, len(self.trainset))
    print >> sys.stderr, 'test set = %s, users = %s' % (testset_len, len(self.testset))
      

if __name__ == '__main__':
  ratingfile = '../dataset/reviews.txt'

  db = DataBase()
  db.generate_dataset(ratingfile)
