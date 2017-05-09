#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from tools.Dataset import DataSet
from tools.utils import ColorPrint

class MatrixFactoriztion:
  def __init__(self, filename):
    self.trainSet, self.testSet, self.userCount, self.itemCount = DataSet(filename).get_dataset()
    
    self.k = 50
    self.learnRate = 0.001
    self.reg_lambda = tf.constant(0.001, dtype=tf.float32)
    self.nStep = 200000
    self.batchSize = 1024
    
    self.itemSet = np.array(list(set(self.trainSet[:, 1])))
    
  def buildGraph(self):
    self.userIdx = tf.placeholder(tf.int32, [None])
    self.itemIdx = tf.placeholder(tf.int32, [None])
    self.ratings = tf.placeholder(tf.float32, [None])
    
    self.userFactor = tf.Variable(tf.truncated_normal([self.userCount, self.k], stddev = 0.001), name = 'UserFactor')
    self.itemFactor = tf.Variable(tf.truncated_normal([self.itemCount, self.k], stddev = 0.001), name = 'ItemFactor')
    self.userBias = tf.Variable(tf.truncated_normal([self.userCount], stddev = 0.001), name = 'UserBias')
    self.itemBias = tf.Variable(tf.truncated_normal([self.itemCount], stddev = 0.001), name = 'ItemBias')
    
    self.userFactorEmbed = tf.nn.embedding_lookup(self.userFactor, self.userIdx)
    self.itemFactorEmbed = tf.nn.embedding_lookup(self.itemFactor, self.itemIdx)
    self.userBiasEmbed = tf.nn.embedding_lookup(self.userBias, self.userIdx)
    self.itemBiasEmbed = tf.nn.embedding_lookup(self.itemBias, self.itemIdx)
    # some questions here: it seems the author want to calculate the seril
    self.ratingMatrix = tf.reduce_sum(tf.multiply(self.userFactorEmbed, self.itemFactorEmbed), 1)
    self.ratingMatrix = tf.add(self.ratingMatrix, self.userBiasEmbed)
    self.ratingMatrix = tf.add(self.ratingMatrix, self.itemBiasEmbed)
    
    self.RMSE = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.ratings, self.ratingMatrix))))
    self.l2_loss = tf.nn.l2_loss(tf.subtract(self.ratings, self.ratingMatrix))
    self.MAE = tf.reduce_mean(tf.abs(tf.subtract(self.ratings, self.ratingMatrix)))
    self.regulazation = tf.add(tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.userFactor)), tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.itemFactor)))
    self.loss = tf.add(self.l2_loss, self.regulazation)
    
    self.optimizer = tf.train.AdamOptimizer(self.learnRate)
    self.trainStepUser = self.optimizer.minimize(self.loss, var_list = [self.userFactor, self.userBias])
    self.trainStepItem = self.optimizer.minimize(self.loss, var_list = [self.itemFactor, self.itemBias])
    
    # evaluation related variable
    self.sUser = tf.placeholder(tf.int32, [None])
    self.sUserFactor = tf.nn.embedding_lookup(self.userFactor, self.sUser)
    self.sUserBias = tf.nn.embedding_lookup(self.userBias, self.sUser)
    self.sRatings = tf.add(tf.add(tf.reduce_sum(tf.matmul(self.sUserFactor, tf.transpose(self.itemFactor)), 0), self.sUserBias), self.itemBias)
    
    # Saver
    tf.summary.scalar("RMSE", self.RMSE)
    tf.summary.scalar("MAE", self.MAE)
    tf.summary.scalar("L2-Loss", self.l2_loss)
    tf.summary.scalar("Reg-Loss", self.loss)
    
    self.summary_op = tf.summary.merge_all()
    
    self.saver = tf.train.Saver()
    
  def recommend(self, user, n = 50):
    hasBought = self.trainSet[np.where(self.trainSet[:, 0] == user)][:, 1]
    nonBought = [item for item in self.itemSet if item not in hasBought]
    feed_dict = {self.sUser: [user], self.sItem: nonBought}
    recRatings = self.sess.run(self.sRatings, feed_dict = feed_dict)
    finalMatrix = np.array(sorted(np.array([nonBought, recRatings]).transpose(), key = lambda x: x[1], reverse = True))
    return finalMatrix[0:n, 0]
    
  def evaluate(self, n = 50):
    coverageSet = set()
    hit, rec_count, test_count = 0, 0, 0
    
    for i in range(10):
      users = list(set(self.trainSet[:, 0]))[(i * self.userCount / 10) : (i + 1) * self.userCount / 10]
      finalMatrix = self.recommend(users)
      
      for user in list(set(self.trainSet[:, 0]))[(i * self.userCount / 10) : (i + 1) * self.userCount / 10]:
        testItems = self.testSet[np.where(self.testSet[:, 0] == user)][:, 1]
        hasBought = self.trainSet[np.where(self.trainSet[:, 0] == user)][:, 1]
        nonBought = [item for item in self.itemSet if item not in hasBought]
        ratingMatrix = finalMatrix[user - i * self.userCount / 10]
        recItems = np.array(sorted(np.array([nonBought, ratingMatrix[nonBought]]).transpose(), key = lambda x: x[1], reverse = True))[0:n, 0]
        for item in recItems:
          if item in testItems: hit += 1
          coverageSet.add(item)
          rec_count += 1
        test_count += len(testItems)
        ColorPrint('Recommend for a user')
    
    precision = hit / (1.0 * rec_count)
    recall = hit / (1.0 * test_count)
    coverage = len(all_rec_movies) / (1.0 * self.movie_count)
    
    ColorPrint('precision=%.4f\trecall=%.4f\tcoverage=%.4f' % (precision, recall, coverage))
  
  def train(self):
    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer())
    
    trainWriter = tf.summary.FileWriter('./log', graph=self.sess.graph)
    for step in range(1, self.nStep):
      batchIdx = np.random.randint(len(self.trainSet), size = self.batchSize)
      feed_dict = {self.userIdx: self.trainSet[batchIdx][:, 0], self.itemIdx: self.trainSet[batchIdx][:, 1], self.ratings: self.trainSet[batchIdx][:, 2]}
      
      self.sess.run(self.trainStepItem, feed_dict = feed_dict)
      _, summaryStr = self.sess.run([self.trainStepUser, self.summary_op], feed_dict = feed_dict)
      
      trainWriter.add_summary(summaryStr, step)
      
      if step % int(self.nStep / 1000) == 0: ColorPrint('RMSE: %f, MAE: %f' % (self.sess.run(self.RMSE, feed_dict = feed_dict), self.sess.run(self.MAE, feed_dict = feed_dict)), 1)
      # if step % int(self.nStep / 100) == 0: self.evaluate()
    
    self.saver.save(self.sess, "log/model.ckpt")
if __name__ == '__main__':
  model = MatrixFactoriztion('./dataset/Ama/reviews.txt')
  model.buildGraph()
  model.train()