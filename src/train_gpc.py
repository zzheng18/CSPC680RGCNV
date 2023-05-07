import time
import tensorflow.compat.v1 as tf
# tf.disable_eager_execution()
tf.config.run_functions_eagerly(True)
tf.enable_eager_execution()
from utils import *
from models import RGCN
import random
import pandas as pd
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from gcn.models import GCN
# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)
random.seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer.')
flags.DEFINE_float('dropout', 0.6, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('para_var', 1, 'Parameter of variance-based attention')
flags.DEFINE_float('para_kl', 5e-4, 'Parameter of kl regularization')
flags.DEFINE_float('para_l2', 5e-4, 'Parameter for l2 loss.')
flags.DEFINE_integer('early_stopping', 20, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, label = load_data(FLAGS.dataset)

# train_mask[641:700] = True

perturbed_features, y_train_gpc = perturb_features_gpc(features, 0.8)
gpc_idx, gpc_feature, gpc_y_train_gpc = get_gpc_train_data(perturbed_features, y_train_gpc, train_mask, test_mask, val_mask)

print(len(gpc_idx))

kernel = 1.0 * RBF(1.0)
gpc = GaussianProcessClassifier(kernel=kernel,
        random_state=0).fit(gpc_feature[:500], gpc_y_train_gpc[:500])
# gpc.score(perturbed_features, y_train_gpc)
gpc_res = gpc.predict(perturbed_features)
# print(type(gpc_res))
# gpc_res = np.asarray(y_train_gpc) # delete later, used for testing

# print(type(y_train_gpc))
gpc_predict_pert_idx = [i for i, x in enumerate(gpc_res==1) if x]

# print(all_idx_to_remove)

features = sp.csr_matrix(perturbed_features)
# features, y_train, train_mask, adj, label, y_val, val_mask = remove_pert(features, y_train, train_mask, adj, label, y_val, val_mask, gpc_predict_pert_idx)
features = modify_pert(features, gpc_predict_pert_idx) # for continuous features
# print(train_mask)

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
model = GCN(placeholders, input_dim=features[2][1], logging=True)
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
    # if epoch == (FLAGS.epochs-1):
    #     # print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    #     df = pd.DataFrame(data = outs[3])
    #     df.to_csv('/home/zihe-leon/Desktop/RobustGCN-master/src/var0.csv', index = False)
    cost, _, duration = evaluate(features, support, y_val, val_mask, placeholders, adj)
    cost_val.append(cost)
    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "time=", "{:.5f}".format(time.time() - t))
    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break
print("Optimization Finished!")

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, label = load_data(FLAGS.dataset)
features = preprocess_features(features)
support = [preprocess_adj(adj, -0.5), preprocess_adj(adj, -1.0)]
test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders, adj)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

