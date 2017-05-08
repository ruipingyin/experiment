#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from tools.Dataset import DataSet
from tools.utils import ColorPrint

class MatrixFactoriztion:
  def __init__(self, filename):
    self.trainSet, self.testSet, self.userCount, self.itemCount = DataSet(filename).get_dataset()
    
    self.k = 20
    self.learnRate = 0.01
    self.reg_lambda = tf.constant(0.01, dtype=tf.float32)
    self.nStep = 10000
    self.batchSize = 100000
    
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
    self.ratingMatrix = tf.reduce_sum(tf.mul(self.userFactorEmbed, self.itemFactorEmbed), 1)
    self.ratingMatrix = tf.add(self.ratingMatrix, self.userBiasEmbed)
    self.ratingMatrix = tf.add(self.ratingMatrix, self.itemBiasEmbed)
    
    self.RMSE = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(self.ratings, self.ratingMatrix))))
    self.l2_loss = tf.nn.l2_loss(tf.sub(self.ratings, self.ratingMatrix))
    self.MAE = tf.reduce_mean(tf.abs(tf.sub(self.ratings, self.ratingMatrix)))
    self.regulazation = tf.add(tf.mul(self.reg_lambda, tf.nn.l2_loss(self.userFactor)), tf.mul(self.reg_lambda, tf.nn.l2_loss(self.itemFactor)))
    self.loss = tf.add(self.l2_loss, self.regulazation)
    
    self.optimizer = tf.train.AdamOptimizer(self.learnRate)
    self.trainStepUser = self.optimizer.minimize(self.loss, var_list = [self.userFactor, self.userBias])
    self.trainStepItem = self.optimizer.minimize(self.loss, var_list = [self.itemFactor, self.itemBias])
    
    # evaluation related variable
    self.sUser = tf.placeholder(tf.int32, 1)
    self.sUserFactor = tf.nn.embedding_lookup(self.userFactor, self.sUser)
    self.sUserBias = tf.nn.embedding_lookup(self.userBias, self.sUser)
    self.sRatings = tf.mul(self.sUser, self.itemBias)
    
  def recommend(self, user, n = 50):
    # except boughted
    self.sUserFactor = tf.nn.embedding_lookup(self.userFactor, self.userIdx)
  
  def evaluate(self):
    
  
  def train(self):
    self.sess = tf.Session()
    self.sess.run(tf.initialize_all_variables())
    
    for step in range(1, self.nStep):
      batchIdx = np.random.randint(len(self.trainSet), size = self.batchSize)
      feed_dict = {self.userIdx: self.trainSet[batchIdx][:, 0], self.itemIdx: self.trainSet[batchIdx][:, 1], self.ratings: self.trainSet[batchIdx][:, 2]}
      
      self.sess.run(self.trainStepItem, feed_dict = feed_dict)
      self.sess.run(self.trainStepUser, feed_dict = feed_dict)
      
      if step % int(self.nStep / 100) == 0: ColorPrint('RMSE: %f, MAE: %f' % (self.sess.run(self.RMSE, feed_dict = feed_dict), self.sess.run(self.MAE, feed_dict = feed_dict)), 1)
    
if __name__ == '__main__':
  model = MatrixFactoriztion('./dataset/Ama/reviews.txt')
  model.buildGraph()
  model.train()