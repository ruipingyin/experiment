import math, random, sys
from operator import itemgetter
from tools.utils import ColorPrint
from tools.Dataset import DataBase

class UserBasedCF():
  def __init__(self, filename, n_sim_user = 200, n_rec_movie = 100):
    self.dataset = DataBase(filename)
    self.n_sim_user, self.n_rec_movie = n_sim_user, n_rec_movie
    self.user_sim_mat = {}
    self.movie_count = 0
    
  def calc_user_sim(self):
    movie2users = {}
    for user, items in self.dataset.trainset.items():
      for movie in items:
        movie2users.setdefault(movie, set())
        movie2users[movie].add(user)
    
    self.movie_count = len(movie2users)
    ColorPrint('total movie number = %d' % self.movie_count, 1)
    
    ColorPrint('building user co-rated movies matrix...', 1)
    for movie, users in movie2users.iteritems():
      for user_a in users:
        for user_b in users:
          if user_a == user_b: continue
          self.user_sim_mat.setdefault(user_a, {})
          self.user_sim_mat[user_a].setdefault(user_b, 0)
          self.user_sim_mat[user_a][user_b] += 1
    
    ColorPrint('calculating user similarity matrix...', 1)
    simfactor_count = 0
    for user, related_users in self.user_sim_mat.iteritems():
      for rUser, count in related_users.iteritems():
        self.user_sim_mat[user][rUser] = count / math.sqrt(len(self.dataset.trainset[user]) * len(self.dataset.trainset[rUser]))
        simfactor_count += 1
        if simfactor_count % 2000000 == 0: ColorPrint('calculating user similarity factor(%d)' % simfactor_count, 1)
        
  def recommend(self, user):
    ''' Find K similar users and recommend N movies. '''
    K = self.n_sim_user
    N = self.n_rec_movie
    
    rank = dict()
    watched_movies = self.dataset.trainset[user]
    
    if user not in self.user_sim_mat: return []
    
    for user, wuv in sorted(self.user_sim_mat[user].items(), key=itemgetter(1), reverse=True)[0:(K if len(self.user_sim_mat[user].items()) > K else len(self.user_sim_mat[user].items()))]:
      for movie in self.dataset.trainset[user]:
        if movie in watched_movies: continue
        rank.setdefault(movie, 0)
        rank[movie] += wuv
    return sorted(rank.items(), key=itemgetter(1), reverse=True)[0:N]
  
  def evaluate(self):
    ColorPrint('Evaluation start...', 1)
    N = self.n_rec_movie
    
    hit, rec_count, test_count = 0, 0, 0
    
    all_rec_movies = set()
    
    for i, user in enumerate(self.dataset.trainset):
      if i % 500 == 0: sys.stderr.write('Done %d / %d\r' % (i, len(self.dataset.testset)))
      if i == len(self.dataset.trainset) - 1: sys.stderr.write(' ' * 50 + '\r')
      test_movies = self.dataset.testset[user]
      rec_movies = self.recommend(user)
      
      for movie, w in rec_movies:
        if movie in test_movies:
          hit += 1
        all_rec_movies.add(movie)
      rec_count += N
      test_count += len(test_movies)
      
    precision = hit / (1.0 * rec_count)
    recall = hit / (1.0 * test_count)
    coverage = len(all_rec_movies) / (1.0 * self.movie_count)
    
    ColorPrint('precision=%.4f\trecall=%.4f\tcoverage=%.4f' % (precision, recall, coverage))
      

if __name__ == '__main__':
  model = UserBasedCF('./dataset/ml_100k.txt')
  model.calc_user_sim()
  model.evaluate()
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
