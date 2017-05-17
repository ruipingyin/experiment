import UserBased
from MatrixFactoriztion import MatrixFactoriztionModel
from MatrixFactoriztionAUC import MatrixFactoriztionAUCModel

''' User-based Collaborative Method (Implicit Rating) '''
def CF_Test():
  model = UserBased.UserBasedCF('./dataset/Ama/reviews.txt')
  model.calc_user_sim()
  model.evaluate()
  
def MF_Explicit_Test():
  model = MatrixFactoriztionModel('./dataset/ml_100k.txt', False)
  model.buildGraph()
  model.train()
  
def MF_Implicit_Test():
  model = MatrixFactoriztionModel('./dataset/ml_100k.txt', True)
  model.buildGraph()
  model.train()
  
def MF_Explicit_Auc_Test():
  model = MatrixFactoriztionAUCModel('./dataset/ml_100k.txt', False)
  model.buildGraph()
  model.train()
  
MF_Explicit_Test()