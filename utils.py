import numpy as np
import torch
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
import scipy.sparse as sp
from tensorflow.python.ops import array_ops

def weight_variable_glorot(input_dim,output_dim,name=""):
    init_range = np.sqrt(6.0/(input_dim+output_dim))
    initial = tf.compat.v1.random_uniform(
        [input_dim,output_dim],
        minval=-init_range,
        maxval=init_range,
        dtype=tf.float32
    )

    return tf.Variable(initial,name=name)


def dropout_sparse(x,keep_prob,num_nonzero_elems):
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor +=tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor),dtype=tf.bool)
    pre_out = tf.sparse_retain(x,dropout_mask)
    return pre_out*(1./keep_prob)

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row,sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords,values,shape

def preprocess_graph(adj):
    adj_ = sp.coo_matrix(adj)
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum,-0.5).flatten())
    adj_nomalized = adj_.dot(
        degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    adj_nomalized = adj_nomalized.tocoo()
    return  sparse_to_tuple(adj_nomalized)

def constructNet(met_dis_matrix):
    met_matrix = np.matrix(
        np.zeros((met_dis_matrix.shape[0],met_dis_matrix.shape[0]),dtype=np.int8))
    dis_matrix = np.matrix(
        np.zeros((met_dis_matrix.shape[1],met_dis_matrix.shape[1]),dtype=np.int8))

    mat1 = np.hstack((met_matrix,met_dis_matrix))
    mat2 = np.hstack((met_dis_matrix.T,dis_matrix))
    adj = np.vstack((mat1,mat2))
    return adj

def constructHNet(met_dis_matrix,met_matrix,dis_matrix):
    mat1 = np.hstack((met_matrix,met_dis_matrix))
    mat2 = np.hstack((met_dis_matrix.T,dis_matrix))
    return np.vstack((mat1,mat2))


