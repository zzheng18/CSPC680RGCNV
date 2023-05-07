import time
import tensorflow.compat.v1 as tf
# tf.disable_eager_execution()
tf.config.run_functions_eagerly(True)
tf.enable_eager_execution()
from utils import *
from models import RGCN
import random
import pandas as pd
# Set random seed
# seed = 123
# np.random.seed(seed)
# tf.set_random_seed(seed)
# random.seed(seed)
from gcn.models import GCN

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden', 32, 'Number of units in hidden layer.')
flags.DEFINE_float('dropout', 0.6, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('para_var', 1, 'Parameter of variance-based attention')
flags.DEFINE_float('para_kl', 5e-4, 'Parameter of kl regularization')
flags.DEFINE_float('para_l2', 5e-4, 'Parameter for l2 loss.')
flags.DEFINE_integer('early_stopping', 20, 'Tolerance for early stopping (# of epochs).')

flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer.')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, label = load_data(FLAGS.dataset)
# add noise to the data
features = perturb_features(features, 0.8) # 0.5, 0.8 is good (cora) # our method is better for higher perturbation
# features = perturb_features_gaussian(features, 0.8) # only use for pubmed
features = preprocess_features(features)

support = [preprocess_adj(adj, -0.5), preprocess_adj(adj, -1.0)]
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(2)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32),
}
model = RGCN(placeholders, input_dim=features[2][1], logging=True)
sess = tf.Session()
def evaluate(features, support, labels, mask, placeholders, adj):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders, adj)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)

sess.run(tf.global_variables_initializer())
cost_val = []
var1 = []
for epoch in range(FLAGS.epochs):
    t = time.time()
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders, adj)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    outs = sess.run([model.opt_op, model.loss, model.accuracy, model.vars], feed_dict=feed_dict)
    # print(outs[3].shape)
    if epoch == (FLAGS.epochs-1):
        # print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        df = pd.DataFrame(data = outs[3])
        df.to_csv('/home/zihe-leon/Desktop/RobustGCN-master/src/var0.csv', index = False)
    cost, _, duration = evaluate(features, support, y_val, val_mask, placeholders, adj)
    cost_val.append(cost)
    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "time=", "{:.5f}".format(time.time() - t))
    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break
print("Optimization Finished?")

_, features, _, _, _, _, _, _, _ = load_data(FLAGS.dataset)
features = preprocess_features(features)
test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders, adj)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

########################################################################################################
var0 = pd.read_csv('/home/zihe-leon/Desktop/RobustGCN-master/src/var0.csv')
var0['mean'] = var0.mean(axis=1)

# calculate thresholds
mean_mean = np.mean(var0['mean'] )
mean_std = np.std(var0['mean'] )
thrs = [mean_mean-2*mean_std, mean_mean-mean_std, mean_mean, mean_mean+mean_std, mean_mean+2*mean_std]

# get a list of index to remove 
rm_pert_idx = get_pert_rm_idx(mean_mean-2*mean_std, mean_mean+2*mean_std, var0) # mean_mean-1*mean_std is good for cora

print(mean_mean-2*mean_std, mean_mean+2*mean_std)
# train RGCN again
print("####################################### Train Again ####################################################")
#################################### original model ######################################################

# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, label = load_data(FLAGS.dataset)
# add noise to the data
cleaned_features, y_train, train_mask, adj, label, y_val, val_mask = remove_pert(features, y_train, train_mask, adj, label, y_val, val_mask, rm_pert_idx)
# cleaned_features = modify_pert(features, rm_pert_idx) # for continuous features
print(y_train.shape[1])
features = preprocess_features(cleaned_features)

support = [preprocess_adj(adj, -0.5), preprocess_adj(adj, -1.0)]
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(2)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32),
}
model = RGCN(placeholders, input_dim=features[2][1], logging=True)
print(features[2][1])
sess = tf.Session()
def evaluate(features, support, labels, mask, placeholders, adj):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders, adj)

    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)

sess.run(tf.global_variables_initializer())
cost_val = []
var1 = []

for epoch in range(FLAGS.epochs):
    t = time.time()
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders, adj)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    outs = sess.run([model.opt_op, model.loss, model.accuracy, model.vars], feed_dict=feed_dict)
    # print(outs[3].shape)
    if epoch == (FLAGS.epochs-1):
        # print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        df = pd.DataFrame(data = outs[3])
        df.to_csv('/home/zihe-leon/Desktop/RobustGCN-master/src/var0_new.csv', index = False)
    
    cost, _, duration = evaluate(features, support, y_val, val_mask, placeholders, adj)
    cost_val.append(cost)
    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "time=", "{:.5f}".format(time.time() - t))
    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break
print("Optimization Finished!")



# Testing
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, label = load_data(FLAGS.dataset)
features = preprocess_features(features)
print(y_test.shape)
support = [preprocess_adj(adj, -0.5), preprocess_adj(adj, -1.0)]
test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders, adj)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

# print(tf.Session().run(tf.constant([1,2,3])))

# print(tf.Session().run(model.vars))


# import tensorflow as tf
# import numpy as np
# tensor = tf.constant(np.arange(1, 5, dtype=np.int32), shape=[2, 2])
# tensor2  = tf.constant(np.arange(1, 5, dtype=np.int32), shape=[2, 2])
# def dot(x, y, sparse=False):

#     if sparse:
#         res = tf.sparse_tensor_dense_matmul(x, y)
#     else:
#         res = tf.matmul(x, y)
#     return res
# eager = tf.nn.relu(dot(tensor, tensor2))
# print(type(eager))
# import tensorflow.compat.v1 as tf
# print(tf.Session().run(eager))

# x = tf.placeholder(tf.int64, shape=[None])

# feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders, adj)
# feed_dict.update({placeholders['dropout']: FLAGS.dropout})

# print(tf.Session().run(model.vars, feed_dict =feed_dict))


# with tf.Session() as sess:  print(model.vars.eval()) 

# print(type())

# print(tf.Session().run(model.vars))

# print(model.vars.eval(session = tf.Session()))
# import tensorflow as tf2

# tf2.print(model.vars, output_stream=sys.stderr)

# tensor = tf2.range(10)
# tf2.print(tensor)
print("~~~~~~~~~~~~~~~~~~~~~~~~")