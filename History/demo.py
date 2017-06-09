#!/usr/bin/python
"""Demo script for running

Authors : Shun Nukui
License : GNU General Public License v2.0

Usage: python demo.py
"""
from __future__ import print_function
import time
import numpy as np
import tensorflow as tf
from tfnmf import TFNMF

from tools.Dataset import DataSet
from tools.utils import ColorPrint

trainSet, testSet, uCount, iCount = DataSet('dataset/ml_100k.txt', False).get_matrix()

def main():
    V = trainSet
    rank = 10

    tfnmf = TFNMF(V, rank)
    with tf.Session() as sess:
        start = time.time()
        W, H = tfnmf.run(sess, max_iter=100000)
        print("Computational Time for TFNMF: ", time.time() - start)

    W = np.mat(W)
    H = np.mat(H)
    
    user_distribution = W
    item_distribution = H

    reconstruct_matrix = np.dot(user_distribution, item_distribution)
    filter_matrix = trainSet < 1e-6

    final_matrix = np.array(reconstruct_matrix) * filter_matrix

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

if __name__ == '__main__':
    main()
