import sys, random, math
from operator import itemgetter

class DataBase():
    def __init__(self):
        self.dataset = {}
        self.trainset = {}
        self.testset = {}

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
        
        trainset_len = 0
        testset_len = 0
        for line in self.loadfile(filename):
            user, item, rating, timestamp = line.split('::')
            
            self.dataset.setdefault(int(user), {})
            self.dataset[int(user)][int(item)] = long(long(timestamp) * 10 + int(float(rating)))
        
        # split the data by pivot
        for user, items in self.dataset.items():
            for item, timestamp in sorted(items.items(), key=itemgetter(1), reverse=True)[0:int(len(items) * (1 - pivot))]:
                self.testset.setdefault(user, {})
                self.testset[user][item] = int(timestamp % 10)
                testset_len = testset_len + 1
            for item, timestamp in sorted(items.items(), key=itemgetter(1), reverse=True)[int(len(items) * (1 - pivot)):]:
                self.trainset.setdefault(user, {})
                self.trainset[user][item] = int(timestamp % 10)
                trainset_len = trainset_len + 1

        print >> sys.stderr, 'train set = %s, users = %s' % (trainset_len, len(self.trainset))
        print >> sys.stderr, 'test set = %s, users = %s' % (testset_len, len(self.testset))
        
        with open('../dataset/trainset.txt', 'w') as trainfile:
            for user, items in self.trainset.items():
                for item, rating in items.items():
                    trainfile.write(str(user))
                    trainfile.write('::')
                    trainfile.write(str(item))
                    trainfile.write('::')
                    trainfile.write(str(rating))
                    trainfile.write('\n')
        with open('../dataset/testset.txt', 'w') as testfile:
            for user, items in self.testset.items():
                for item, rating in items.items():
                    testfile.write(str(user))
                    testfile.write('::')
                    testfile.write(str(item))
                    testfile.write('::')
                    testfile.write(str(rating))
                    testfile.write('\n')
        


ratingfile = '../dataset/reviews.txt'

db = DataBase()
db.generate_dataset(ratingfile)