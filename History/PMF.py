#!/usr/bin/env python

from __future__ import division

import numpy as np
from scipy import sparse
import sys
from tools.Dataset import DataSet
from tools.utils import ColorPrint


np.seterr(all='raise')

class PMF:

    def __init__(self, file_path, f, gamma=0.005, lambda_=0.02, min_iter = 1, max_iter=1000, 
                 min_improvement=1e-4, p=None, q=None):
        trainSet, self.testSet, uCount, iCount = DataSet('dataset/ml_100k.txt', False).get_matrix()
        
        self.R_sparse = self._load_data(trainSet)
        print self.R_sparse.shape, uCount, iCount
        self.filter_matrix = trainSet < 1e-6
        self.R = self.R_sparse.todense()
        self.f = f
        self.gamma = gamma
        self.lambda_ = lambda_
        self.K_users, self.K_items = self.R_sparse.nonzero()

        # self.n, self.m depend on R_sparse user, item vectors starting at 0
        self.n = uCount
        self.m = iCount

        # load low-rank matrices `p` (item) or `q` (user) matrix
        #   otherwise, randomly initalize
        if hasattr(self, 'p'):
            self.p = p
        else:
            self.p = np.ones((self.f, self.n)) * np.random.uniform(-0.01, 0.01,
                                                     size=self.f*self.n).reshape(self.f,self.n)
        if hasattr(self, 'q'):
            self.q = q
        else:
            self.q = np.ones((self.f, self.m)) * np.random.uniform(-0.01, 0.01,
                                                     size=self.m*self.f).reshape(self.f,self.m)

        self.min_iter = min_iter
        self.max_iter = max_iter
        self.min_improvement = min_improvement

        # initialize average rating, and initial user/item biases
        self.get_baseline()


    def get_rating(self, u, i):
        return self.R[u,i]


    def predict_rhat(self, u, i):
        r_hat = self.mu + self.b_i[i] + self.b_u[u] + \
                     np.dot(self.q.T[i,:], self.p[:,u])
        if r_hat > 5:
            return 5
        elif r_hat < 1:
            return 1
        else:
            return r_hat


    def _get_mu(self, x):
        return np.nansum(x) / np.count_nonzero(~np.isnan(x)) or self._get_mu(self.R)


    def get_baseline(self):
        self.mu = self._get_mu(self.R)
        self.b_u = self.mu - [self._get_mu(self.R[u,:]) for u in xrange(self.n)]
        self.b_i = self.mu - [self._get_mu(self.R[:,i]) for i in xrange(self.m)]
        return None


    def update(self, u, i, f,  err):
        self.b_u[u] = self.b_u[u] + self.gamma * (err - self.lambda_ * self.b_u[u])
        self.b_i[i] = self.b_i[i] + self.gamma * (err - self.lambda_ * self.b_i[i])
        self.q[f,i] = self.q[f,i] + self.gamma * (err * self.p[f,u] - self.lambda_ * self.q[f,i])
        self.p[f,u] = self.p[f,u] + self.gamma * (err * self.q[f,i] - self.lambda_ * self.p[f,u])
        return None
    

    def compute_cost(self, u, i):
        r = self.get_rating(u, i)
        r_hat_desc = self.mu - self.b_i[i] - self.b_u[u] - np.dot(self.q.T[i,:], self.p[:,u])
        cost = (r - r_hat_desc)**2 + self.lambda_ * \
               (self.b_i[i]**2 + self.b_u[u]**2 + np.linalg.norm(self.q[:,i])**2 + \
                                           np.linalg.norm(self.p[:,u])**2)
        return cost


    def get_error(self, u, i):
        r = self.get_rating(u, i)
        r_hat = self.predict_rhat(u, i)
        err = r - r_hat
        return err


    def train(self):
        cost = 2.0
        n_ratings = len(self.K_items)
        self.sse = 0

        for feature in xrange(self.f):
            print('--- Calculating Feature: {feat} ---'.format(feat=feature + 1))

            for n in xrange(self.max_iter):
                cost_last = cost

                for u, i in zip(self.K_users, self.K_items): 
                    err = self.get_error(u, i)
                    self.update(u, i, feature, err)
                    self.sse += err**2
                        
                cost = self.compute_cost(u, i)

                if (n >= self.min_iter and cost > cost_last - self.min_improvement):
                    break

        self.rmse = np.sqrt(self.sse / n_ratings)
        
        user_distribution, item_distribution = np.transpose(self.p), self.q
        reconstruct_matrix = np.dot(user_distribution, item_distribution)
        final_matrix = reconstruct_matrix * self.filter_matrix
        
        coverageSet = set()
        hit, rec_count, test_count = 0, 0, 0

        for user in range(0, self.n):
            testItems = self.testSet[np.where(self.testSet[:, 0] == user)][:, 1]
            recItems = np.argpartition(final_matrix[user], -50)[-50:]
            for item in recItems:
                if item in testItems: hit += 1
                coverageSet.add(item)
                rec_count += 1
            test_count += len(testItems)
          
        precision = hit / (1.0 * rec_count)
        recall = hit / (1.0 * test_count)
        coverage = len(coverageSet) * 1.0 / self.m
        ColorPrint('precision=%.4f\trecall=%.4f\tcoverage=%.4f' % (precision, recall, coverage))


    @staticmethod
    def _load_data(trainSet):
        '''
        Accepts a tab-delimited file with first three columns as
            user, item, value.
        Returns sparse matrix.
        '''
        data = np.array(trainSet)
        ij = data[:, :2]
        # ij -= 1
        values = data[:, 2]
        sparse_matrix = sparse.csc_matrix(data).astype(float)

        # sparse_matrix = sparse.csc_matrix((values, ij.T)).astype(float)
        return sparse_matrix


if __name__ == '__main__':
    test = PMF('./dataset/ml_100k.txt', 25)
    test.train()
