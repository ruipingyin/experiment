#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from tools.Dataset import DataSet
from tools.utils import ColorPrint
from time import clock

INFINITY = 10e+16

class MatrixFactoriztionModel:
  def __init__(self, filename, implicit):
    self.trainSet, self.testSet, self.userCount, self.itemCount = DataSet(filename, implicit).get_dataset()
    
    self.latentDim = 10
    self.learnRate = 0.005
    self.nStep = 20000
    self.batchSize = 512
    
    self.itemSet = range(0, self.itemCount)
    
  def buildGraph(self):
    self.userIdx = tf.placeholder(tf.int32, [None])
    self.itemIdx = tf.placeholder(tf.int32, [None])
    self.ratings = tf.placeholder(tf.float32, [None])
    
    scale = 2 * np.sqrt(2.5 / self.latentDim)
    
    self.userFactor = tf.Variable(tf.random_uniform([self.userCount, self.latentDim], maxval=scale), name = 'UserFactor')
    self.itemFactor = tf.Variable(tf.random_uniform([self.itemCount, self.latentDim], maxval=scale), name = 'ItemFactor')
    # self.userBias = tf.Variable(tf.random_uniform([self.userCount], maxval=scale), name = 'UserBias')
    # self.itemBias = tf.Variable(tf.random_uniform([self.itemCount], maxval=scale), name = 'ItemBias')
    
    # self.gloBias = tf.Variable(tf.random_uniform([1], maxval=scale), name = 'GlobalBias')
    
    self.userFactorEmbed = tf.nn.embedding_lookup(self.userFactor, self.userIdx)
    self.itemFactorEmbed = tf.nn.embedding_lookup(self.itemFactor, self.itemIdx)
    # self.userBiasEmbed = tf.nn.embedding_lookup(self.userBias, self.userIdx)
    # self.itemBiasEmbed = tf.nn.embedding_lookup(self.itemBias, self.itemIdx)
    
    self.ratingMatrix = tf.reduce_sum(tf.multiply(self.userFactorEmbed, self.itemFactorEmbed), 1)
    # self.ratingMatrix = tf.add(self.ratingMatrix, self.userBiasEmbed)
    # self.ratingMatrix = tf.add(self.ratingMatrix, self.itemBiasEmbed)
    # self.ratingMatrix = tf.add(self.ratingMatrix, self.gloBias)
    
    # self.RMSE = tf.sqrt(tf.losses.mean_squared_error(self.ratings, self.ratingMatrix))
    # self.l2_loss = tf.nn.l2_loss(tf.subtract(self.ratings, self.ratingMatrix))
    # self.MAE = tf.reduce_mean(tf.abs(tf.subtract(self.ratings, self.ratingMatrix)))
    
    # non negative constraint
    # nn_w = tf.reduce_sum(tf.abs(self.userFactorEmbed) - self.userFactorEmbed)
    # nn_h = tf.reduce_sum(tf.abs(self.itemFactorEmbed) - self.itemFactorEmbed)
    # nn_wb = tf.reduce_sum(tf.abs(self.userBiasEmbed) - self.userBiasEmbed)
    # nn_hb = tf.reduce_sum(tf.abs(self.itemBiasEmbed) - self.itemBiasEmbed)
    # self.constraint = tf.multiply(INFINITY, tf.add(tf.add(nn_w, nn_h), tf.add(nn_wb, nn_hb)))
    # self.loss = tf.add(self.l2_loss, self.constraint)
    
    # self.optimizer = tf.train.AdamOptimizer(self.learnRate).minimize(self.loss)
    #self.trainStepUser = self.optimizer.minimize(self.loss, var_list = [self.userFactor, self.userBias])
    #self.trainStepItem = self.optimizer.minimize(self.loss, var_list = [self.itemFactor, self.itemBias])
    
    # evaluation related variable
    self.sUser = tf.placeholder(tf.int32, [None])
    self.sUserFactor = tf.nn.embedding_lookup(self.userFactor, self.sUser)
    self.sUserBias = tf.nn.embedding_lookup(self.userBias, self.sUser)
    self.sRatings = tf.add(tf.matmul(self.sUserFactor, tf.transpose(self.itemFactor)), self.itemBias)
    
    # Saver
    tf.summary.scalar("RMSE", self.RMSE)
    tf.summary.scalar("MAE", self.MAE)
    tf.summary.scalar("L2-Loss", self.l2_loss)
    tf.summary.scalar("Constraint", self.constraint)
    tf.summary.scalar("Reg-Loss", self.loss)
    
    self.summary_op = tf.summary.merge_all()
    
    self.saver = tf.train.Saver()
    
  def recommend(self, users):
    feed_dict = {self.sUser: users}
    recRatings = self.sess.run(self.sRatings, feed_dict = feed_dict)
    return recRatings
    
  def evaluate(self, n = 50):
    coverageSet = set()
    hit, rec_count, test_count = 0, 0, 0
    
    ColorPrint('User Count: %d' % self.userCount)
    
    batch = 100
    
    for i in range(batch):
      users = list(set(self.trainSet[:, 0]))[(i * self.userCount / batch) : (i + 1) * self.userCount / batch]
      finalMatrix = self.recommend(users)
      
      for user in list(set(self.trainSet[:, 0]))[(i * self.userCount / batch) : (i + 1) * self.userCount / batch]:
        testItems = self.testSet[np.where(self.testSet[:, 0] == user)][:, 1]
        hasBought = self.trainSet[np.where(self.trainSet[:, 0] == user)][:, 1]
        nonBought = np.delete(self.itemSet, hasBought)
        ratingMatrix = finalMatrix[user - i * self.userCount / batch]
        recItems = nonBought[np.argpartition(ratingMatrix[nonBought], -n)[-n:]]
        for item in recItems:
          if item in testItems: hit += 1
          coverageSet.add(item)
          rec_count += 1
        test_count += len(testItems)
      ColorPrint('Processed: %6d, precision=%.4f(%d)\trecall=%.4f\tcoverage=%.4f' % ((i + 1) * self.userCount / batch, hit / (1.0 * rec_count), hit, hit / (1.0 * test_count), len(coverageSet) / (1.0 * self.itemCount)))
    precision = hit / (1.0 * rec_count)
    recall = hit / (1.0 * test_count)
    coverage = len(coverageSet) / (1.0 * self.itemCount)
    
    ColorPrint('precision=%.4f\trecall=%.4f\tcoverage=%.4f' % (precision, recall, coverage))
  
  def train(self):
    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer())
    
    # self.saver.restore(self.sess, './log/model.ckpt')
    # self.evaluate()
    # exit()
    trainWriter = tf.summary.FileWriter('./log/train', graph=self.sess.graph)
    testWriter = tf.summary.FileWriter('./log/test', graph=self.sess.graph)
    
    for step in range(1, self.nStep):
      batchIdx = np.random.randint(len(self.trainSet), size = self.batchSize)
      feed_dict = {self.userIdx: self.trainSet[batchIdx][:, 0], self.itemIdx: self.trainSet[batchIdx][:, 1], self.ratings: self.trainSet[batchIdx][:, 2]}
      
      # self.sess.run(self.trainStepItem, feed_dict = feed_dict)
      _, summaryStr = self.sess.run([self.optimizer, self.summary_op], feed_dict = feed_dict)
      
      trainWriter.add_summary(summaryStr, step)
      
      if step % int(self.nStep / 100) == 0:
        rmse, mae, cst = self.sess.run([self.RMSE, self.MAE, self.constraint], feed_dict = feed_dict)
        ColorPrint('TRIAN-RMSE: %f, TRIAN-MAE: %f, Constraint: %f' % (rmse, mae, cst), 1)
        feed_dict_test = {self.userIdx: self.testSet[:, 0], self.itemIdx: self.testSet[:, 1], self.ratings: self.testSet[:, 2]}
        ColorPrint('TEST- RMSE: %f, TEST- MAE: %f' % (self.sess.run(self.RMSE, feed_dict = feed_dict_test), self.sess.run(self.MAE, feed_dict = feed_dict_test)), 1)
        summaryStr = self.sess.run(self.summary_op, feed_dict = feed_dict_test)
        testWriter.add_summary(summaryStr, step)
      # if step % int(self.nStep / 1000) == 0: self.evaluate()
    self.evaluate()
    self.saver.save(self.sess, "log/model.ckpt")

if __name__ == '__main__':
  model = MatrixFactoriztionModel('./dataset/ml_100k.txt', False)
  model.buildGraph()
  model.train()
