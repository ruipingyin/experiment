from tools.Dataset import DataBase
from tools.utils import ColorPrint

class Model:

  def __init__(self, filename):
    ds = DataBase(filename)
    
    self.trainset, self.testset = ds.get_dataset()
    self.itemCount = ds.get_item_count()
    self.userCount = ds.get_user_count()
    self.ratingCount = ds.get_rating_count()
    
  def recommend(self, user):
    return []
  
  def evaluate(self):
    ColorPrint('Evaluation start...', 1)
    N = self.n_rec_movie
    
    hit, rec_count, test_count = 0, 0, 0
    
    all_rec_movies = set()
    
    popular_sum = 0
    
    for i, user in enumerate(self.trainset):
      if i % 500 == 0: sys.stderr.write('Done %d / %d\r' % (i, len(self.testset)))
      if i == len(self.trainset) - 1: sys.stderr.write(' ' * 50 + '\r')
      test_movies = self.testset[user]
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
  