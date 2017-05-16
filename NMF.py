import numpy as np
from sklearn.decomposition import NMF
from tools.Dataset import DataSet
from tools.utils import ColorPrint

trainSet, testSet, uCount, iCount = DataSet('dataset/ml_100k.txt', False).get_matrix()

nmf = NMF(n_components = 25)
user_distribution = nmf.fit_transform(trainSet)  
item_distribution = nmf.components_

reconstruct_matrix = np.dot(user_distribution, item_distribution)
filter_matrix = trainSet < 1e-6

final_matrix = reconstruct_matrix * filter_matrix

coverageSet = set()
hit, rec_count, test_count = 0, 0, 0

for user in range(0, uCount):
  testItems = testSet[np.where(testSet[:, 0] == user)][:, 1]
  recItems = np.argpartition(final_matrix[user], -50)[-50:]
  for item in recItems:
    if item in testItems: hit += 1
    coverageSet.add(item)
    rec_count += 1
  test_count += len(testItems)
  ColorPrint('precision=%.4f(%d)\trecall=%.4f\tcoverage=%.4f' % (hit / (1.0 * rec_count), hit, hit / (1.0 * test_count), len(coverageSet) / (1.0 * iCount)))
  
precision = hit / (1.0 * rec_count)
recall = hit / (1.0 * test_count)
coverage = len(coverageSet) * 1.0 / iCount
ColorPrint('precision=%.4f\trecall=%.4f\tcoverage=%.4f' % (precision, recall, coverage))