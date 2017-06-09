import UserBased
from MatrixFactoriztion import MatrixFactoriztionModel

filename = './dataset/Ama/reviews.txt'
# filename = './dataset/ml_100k.txt'

''' User-based Collaborative Method (Implicit Rating) '''
def MemGridSearch():
  model = UserBased.UserBasedCF(filename)
  model.calc_user_sim()
  model.evaluate()
  
def MfExplicitGridSearch():
  latentDims = [5, 10, 25, 50, 100, 150, 200]
  learnRates = [0.001, 0.0025, 0.005, 0.01]
  reg_lambdas = [0.001, 0.01, 0.1, 0.5]
  normalStds = [0.01, 0.001]
  batchSizes = [64, 512]
  
  for latentDim in latentDims:
    for learnRate in learnRates:
      for reg_lambda in reg_lambdas:
        for normalStd in normalStds:
          for batchSize in batchSizes:
            model = MatrixFactoriztionModel(filename, False)
            model.latentDim = latentDim
            model.learnRate = learnRate
            model.userNormalStd, model.userBiasStd = normalStd, normalStd
            model.itemNormalStd, model.itemBiasStd = normalStd, normalStd
            model.batchSize = batchSize
            model.regLambda = reg_lambda
            model.buildGraph()
            precision, recall, coverage = model.train()
            with open('./log/gridAma.txt', 'a') as out:
              out.write('%4d, %.6f, %.6f, %.6f, %4d -> precision=%.4f\trecall=%.4f\tcoverage=%.4f\n' % (latentDim, learnRate, reg_lambda, normalStd, batchSize, precision, recall, coverage))
              
  
def MF_Implicit_Test():
  model = MatrixFactoriztionModel(filename, True)
  model.buildGraph()
  model.train()
  
def MF_Explicit_Auc_Test():
  from MatrixFactoriztionAUC import MatrixFactoriztionAUCModel
  
  model = MatrixFactoriztionAUCModel(filename, False)
  model.buildGraph()
  model.train()
  
MfExplicitGridSearch()