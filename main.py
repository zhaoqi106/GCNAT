import numpy as np
import scipy.sparse as sp
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.framework import  ops
ops.reset_default_graph()
import gc
import random
from layer import *
from metric import cv_model_evaluate
from utils import *
from model import GCNModel
from opt import Optimizer

def PredictScore(train_met_dis_matrix, met_matrix, dis_matrix, seed, epochs, emb_dim,dp,lr,adjdp):
    np.random.seed(seed)
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.set_random_seed(seed)
    adj = constructHNet(train_met_dis_matrix, met_matrix, dis_matrix)
    adj = sp.csc_matrix(adj)
    association_nam= train_met_dis_matrix.sum()
    X = constructNet(train_met_dis_matrix)
    features = sparse_to_tuple(sp.csc_matrix(X))
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]
    adj_orig = train_met_dis_matrix.copy()
    adj_orig = sparse_to_tuple(sp.csc_matrix(adj_orig))

    adj_norm = preprocess_graph(adj)
    adj_nonzero = adj_norm[1].shape[0]
    placeholders = {
        'features':tf.compat.v1.sparse_placeholder(tf.float32),
        'adj':tf.compat.v1.sparse_placeholder(tf.float32),
        'adj_orig':tf.compat.v1.sparse_placeholder(tf.float32),
        'dropout':tf.compat.v1.placeholder_with_default(0.,shape=()),
        'adjdp':tf.compat.v1.placeholder_with_default(0.,shape=())
    }
    model = GCNModel(placeholders,num_features,emb_dim,
                     features_nonzero,adj_nonzero,train_met_dis_matrix.shape[0],name='GCNGAT')
    with tf.name_scope('optimizer'):
        opt = Optimizer(
            preds=model.reconstructions,
            labels=tf.reshape(tf.sparse_tensor_to_dense(
                placeholders['adj_orig'], validate_indices=False), [-1]),
            model=model,
            lr=lr,num_u=train_met_dis_matrix.shape[0],num_v=train_met_dis_matrix.shape[1],association_nam=association_nam)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        feed_dict = dict()
        feed_dict.update({placeholders['features']: features})
        feed_dict.update({placeholders['adj']: adj_norm})
        feed_dict.update({placeholders['adj_orig']: adj_orig})
        feed_dict.update({placeholders['dropout']: dp})
        feed_dict.update({placeholders['adjdp']: adjdp})
        _, avg_cost = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)
        if epoch % 100 == 0:
            feed_dict.update({placeholders['dropout']: 0})
            feed_dict.update({placeholders['adjdp']: 0})
            res = sess.run(model.reconstructions, feed_dict=feed_dict)
            print("Epoch:", '%04d' % (epoch + 1),
                  "train_loss=", "{:.5f}".format(avg_cost))
    print('Optimization Finished!')
    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['adjdp']: 0})
    res = sess.run(model.reconstructions, feed_dict=feed_dict)
    sess.close()
    return res

def cross_validation_experiment(met_dis_matrix, met_matrix, dis_matrix, seed, epochs, emb_dim, dp, lr, adjdp):
    #进行交叉验证
    index_matrix = np.mat(np.where(met_dis_matrix == 1))
    association_nam = index_matrix.shape[1]
    random_index = index_matrix.T.tolist()
    random.seed(seed)
    random.shuffle(random_index)
    k_folds = 5
    CV_size = int(association_nam / k_folds)
    temp = np.array(random_index[:association_nam - association_nam %
                                 k_folds]).reshape(k_folds, CV_size,  -1).tolist()
    temp[k_folds - 1] = temp[k_folds - 1] + \
        random_index[association_nam - association_nam % k_folds:]
    random_index = temp
    metric = np.zeros((1, 7))
    print("seed=%d, evaluating met-disease...." % (seed))
    for k in range(k_folds):
        print("------this is %dth cross validation------" % (k+1))
        train_matrix = np.matrix(met_dis_matrix, copy=True)
        train_matrix[tuple(np.array(random_index[k]).T)] = 0
        met_len = met_dis_matrix.shape[0]
        dis_len = met_dis_matrix.shape[1]
        met_disease_res = PredictScore(
            train_matrix, met_matrix, dis_matrix, seed, epochs, emb_dim, dp, lr,  adjdp)
        predict_y_proba = met_disease_res.reshape(met_len, dis_len)
        metric_tmp = cv_model_evaluate(
            met_dis_matrix, predict_y_proba, train_matrix)
        print(metric_tmp)
        metric += metric_tmp
        del train_matrix
        gc.collect()
    print(metric / k_folds)
    metric = np.array(metric / k_folds)
    return metric

if __name__ == "__main__":
    met_sim = np.loadtxt('../data/GM2.csv', delimiter=',')
    dis_sim = np.loadtxt('../data/DSS.csv', delimiter=',')
    met_dis_matrix = np.loadtxt('../data/newdata.csv', delimiter=',')
    epoch = 500
    emb_dim = 64
    lr = 0.01
    adjdp = 0.5
    dp = 0.5
    simw = 2
    result = np.zeros((1, 7), float)
    average_result = np.zeros((1, 7), float)
    circle_time = 1
    for i in range(circle_time):
        result += cross_validation_experiment(
            met_dis_matrix, met_sim*simw, dis_sim*simw, i, epoch, emb_dim, dp, lr, adjdp)
    average_result = result / circle_time
    print(average_result)
