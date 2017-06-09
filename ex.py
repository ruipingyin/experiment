import UserBased
from MatrixFactoriztion import MatrixFactoriztionModel
from tools.utils import ColorPrint


filename = './dataset/Ama/reviews_1.txt'
# filename = './dataset/ml_100k.txt'

''' User-based Collaborative Method (Implicit Rating) '''
def MemGridSearch():
  model = UserBased.UserBasedCF(filename)
  model.calc_user_sim()
  model.evaluate()
  
def MfExplicitGridSearch():
  latentDims = [50] # [700] # [25, 50, 100, 150, 200, 650, 750, 1000]
  learnRates = [0.005] # [0.002] # [0.001, 0.0025, 0.005, 0.01]
  reg_lambdas = [1]  # [0.001] # [0.001, 0.01, 0.1, 0.5, 1, 5, 10]
  normalStds = [0.01] # [0.001] # [0.01, 0.001]
  batchSizes = [128] # [4, 16, 64, 256, 1024, 4096]  # [4096] # [32, 64, 128, 256]
  
  
  with open('./log/gridAma.txt', 'w') as out:
    out.write("Start LatentDim, LearnRate, RegLambda, NormalStd, BatchSize\n")
    for latentDim in latentDims:
      for learnRate in learnRates:
        for reg_lambda in reg_lambdas:
          for normalStd in normalStds:
            for batchSize in batchSizes:
              model = MatrixFactoriztionModel(filename, False, True)
              model.latentDim = latentDim
              model.learnRate = learnRate
              model.userNormalStd, model.userBiasStd = normalStd, normalStd
              model.itemNormalStd, model.itemBiasStd = normalStd, normalStd
              model.batchSize = batchSize
              model.regLambda = reg_lambda
              model.buildGraph()
              # model.train()
              precision, recall, coverage = model.train()
              out.write('%4d, %.6f, %.6f, %.6f, %4d, %.4f, %.4f, %.4f\n' % (latentDim, learnRate, reg_lambda, normalStd, batchSize, precision, recall, coverage))
              ColorPrint('%d, %.6f, %.6f, %.6f, %d -> precision=%.4f\trecall=%.4f\tcoverage=%.4f\n' % (latentDim, learnRate, reg_lambda, normalStd, batchSize, precision, recall, coverage), 1)
              ColorPrint(model.trainSet[1])
  
def MF_Implicit_Test():
  model = MatrixFactoriztionModel(filename, True, True)
  model.buildGraph()
  model.train()
  
def MF_Explicit_Auc_Test():
  from MatrixFactoriztionAUC import MatrixFactoriztionAUCModel
  
  model = MatrixFactoriztionAUCModel(filename, False)
  model.buildGraph()
  model.train()
  
def DataPrep():
  from tools.Dataset import DataModel

  model = DataModel(filename = filename, implicit = False, pivot = 0.7)

  model.dataInfo()

  # model.dataPrep()

  # model = DataModel(filename = './dataset/Ama/reviews_t20.txt', implicit = False, pivot = 0.7)

  # model.dataInfo()
  
  
MfExplicitGridSearch()
# DataPrep()
