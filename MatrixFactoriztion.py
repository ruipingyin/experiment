#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from tools.Dataset import DataModel
from tools.utils import ColorPrint
import sys

class MatrixFactoriztionModel:
  def __init__(self, filename, implicit, AUC):
    self.AUC = AUC
    if self.AUC:
      self.trainSet, self.testSet, self.userCount, self.itemCount = DataModel(filename = filename, implicit = implicit, pivot = -1).dataList()
    else:
      self.trainSet, self.testSet, self.userCount, self.itemCount = DataModel(filename = filename, implicit = implicit, pivot = 0.7).dataList()
    tf.reset_default_graph()
    
    self.latentDim = 30
    self.learnRate = 0.005
    self.userNormalStd, self.userBiasStd = 0.001, 0.001
    self.itemNormalStd, self.itemBiasStd = 0.001, 0.001
    self.nStep = 10000
    self.batchSize = 512
    self.regLambda = 0.1
    self.itemSet = range(0, self.itemCount)
    
  def buildGraph(self):
    self.reg_lambda = tf.constant(self.regLambda, dtype=tf.float32)
  
    self.userIdx = tf.placeholder(tf.int32, [None])
    self.itemIdx = tf.placeholder(tf.int32, [None])
    self.ratings = tf.placeholder(tf.float32, [None])
    
    self.userFactor = tf.Variable(tf.truncated_normal([self.userCount, self.latentDim], stddev = 0.001), name = 'UserFactor')
    self.itemFactor = tf.Variable(tf.truncated_normal([self.itemCount, self.latentDim], stddev = 0.001), name = 'ItemFactor')
    self.userBias = tf.Variable(tf.truncated_normal([self.userCount], stddev = 0.001), name = 'UserBias')
    self.itemBias = tf.Variable(tf.truncated_normal([self.itemCount], stddev = 0.001), name = 'ItemBias')
    
    self.gloBias = tf.Variable(tf.truncated_normal([1], stddev = 0.001), name = 'GlobalBias')
    
    self.userFactorEmbed = tf.nn.embedding_lookup(self.userFactor, self.userIdx)
    self.itemFactorEmbed = tf.nn.embedding_lookup(self.itemFactor, self.itemIdx)
    self.userBiasEmbed = tf.nn.embedding_lookup(self.userBias, self.userIdx)
    self.itemBiasEmbed = tf.nn.embedding_lookup(self.itemBias, self.itemIdx)
    self.ratingMatrix = tf.reduce_sum(tf.multiply(self.userFactorEmbed, self.itemFactorEmbed), 1)
    self.ratingMatrix = tf.add(self.ratingMatrix, self.userBiasEmbed)
    self.ratingMatrix = tf.add(self.ratingMatrix, self.itemBiasEmbed)
    self.ratingMatrix = tf.add(self.ratingMatrix, self.gloBias)
    
    self.RMSE = tf.sqrt(tf.losses.mean_squared_error(self.ratings, self.ratingMatrix))
    self.l2_loss = tf.nn.l2_loss(tf.subtract(self.ratings, self.ratingMatrix))
    self.MAE = tf.reduce_mean(tf.abs(tf.subtract(self.ratings, self.ratingMatrix)))
    self.regulazation1 = tf.add(tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.userFactor)), tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.itemFactor)))
    self.regulazation2 = tf.add(tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.userBias)), tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.itemBias)))
    self.regulazation = tf.add(self.regulazation1, self.regulazation2)
    self.loss = tf.add(self.l2_loss, self.regulazation)
    
    self.optimizer = tf.train.AdamOptimizer(self.learnRate)
    self.trainStepUser = self.optimizer.minimize(self.loss, var_list = [self.userFactor, self.userBias])
    self.trainStepItem = self.optimizer.minimize(self.loss, var_list = [self.itemFactor, self.itemBias])
    
    # evaluation related variable
    self.sUser = tf.placeholder(tf.int32, [None])
    self.sUserFactor = tf.nn.embedding_lookup(self.userFactor, self.sUser)
    self.sUserBias = tf.nn.embedding_lookup(self.userBias, self.sUser)
    self.sRatings = tf.add(tf.matmul(self.sUserFactor, tf.transpose(self.itemFactor)), self.itemBias)
    
    # Saver
    tf.summary.scalar("RMSE", self.RMSE)
    tf.summary.scalar("MAE", self.MAE)
    tf.summary.scalar("L2-Loss", self.l2_loss)
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
    return (precision, recall, coverage)
  
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
      
      self.sess.run(self.trainStepItem, feed_dict = feed_dict)
      _, summaryStr = self.sess.run([self.trainStepUser, self.summary_op], feed_dict = feed_dict)
      
      trainWriter.add_summary(summaryStr, step)
      
      if step % int(self.nStep / 100) == 0:
        ColorPrint('TRIAN-RMSE: %f, TRIAN-MAE: %f' % (self.sess.run(self.RMSE, feed_dict = feed_dict), self.sess.run(self.MAE, feed_dict = feed_dict)), 1)
        feed_dict_test = {self.userIdx: self.testSet[:, 0], self.itemIdx: self.testSet[:, 1], self.ratings: self.testSet[:, 2]}
        ColorPrint('TEST- RMSE: %f, TEST- MAE: %f' % (self.sess.run(self.RMSE, feed_dict = feed_dict_test), self.sess.run(self.MAE, feed_dict = feed_dict_test)), 1)
        summaryStr = self.sess.run(self.summary_op, feed_dict = feed_dict_test)
        testWriter.add_summary(summaryStr, step)
      # if step % int(self.nStep / 1000) == 0: self.evaluate()
    self.saver.save(self.sess, "log/model.ckpt")
    if self.AUC:
      return self.evalAUC()
    else:
      return self.evaluate()

  def evalAUC(self):
    aucList = []
    batch = 100
    
    for i in range(batch):
      users = list(set(self.trainSet[:, 0]))[(i * self.userCount / batch) : (i + 1) * self.userCount / batch]
      finalMatrix = self.recommend(users)
      
      for user in users:
        testItem = self.testSet[np.where(self.testSet[:, 0] == user)][0, 1]
        hasBought = self.trainSet[np.where(self.trainSet[:, 0] == user)][:, 1]
        nonBought = np.delete(self.itemSet, hasBought)
        ratingList = finalMatrix[user - i * self.userCount / batch]
        max, countTest = 0, 0
        for item in range(self.itemCount):
          if item == testItem: continue
          if item in hasBought: continue
          max += 1
          if ratingList[testItem] > ratingList[item]: countTest += 1
          
        aucList.append(1.0 * countTest / max)
      sys.stderr.write('Done %d / %d\r' % (i * batch, self.userCount))
  
    aucValue = np.sum(aucList) * 1.0 / self.userCount
    variance = np.sqrt(np.sum((aucList - aucValue) ** 2) / self.userCount)
    
    ColorPrint('\nAUC=%.4f\tVariance=%.4f' % (aucValue, variance), 1)
