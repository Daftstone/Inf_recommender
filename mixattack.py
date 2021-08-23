import warnings

warnings.filterwarnings("ignore")
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import copy

from Dataset import Dataset
import utils
from model.AutoRec import AutoRec
from model.SVD import SVD
from model.BPR import BPR
from model.CDAE import CDAE
from model.PMF import PMF
from model.NeuMF import NeuMF
from model.SLIM import SLIM
from model.LRec import LRec
from model.NMF import NMF
from model.VAE_CF import VAE_CF
from model.SlopeOne import Slopeone

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("dataset", "ml-100k", "Choose a dataset.")
flags.DEFINE_string('path', 'Data/', 'Input data path.')
flags.DEFINE_string('gpu', '0', 'Input data path.')
flags.DEFINE_integer('batch_size', 64, 'batch_size')
flags.DEFINE_integer('epochs', 20, 'Number of epochs.')
flags.DEFINE_integer('embed_size', 64, 'Embedding size.')
flags.DEFINE_integer('dns', 0, 'number of negative sample for each positive in dns.')
flags.DEFINE_integer('per_epochs', 1, 'pass')
flags.DEFINE_bool('reg_data', True, 'Regularization for adversarial loss')
flags.DEFINE_string('rs', 'pmf', 'recommender system')
flags.DEFINE_string('local_rs', 'pmf', 'recommender system')
flags.DEFINE_bool("is_train", False, "train online or load model")
flags.DEFINE_integer("top_k", 10, "top k")
flags.DEFINE_integer("attack_top_k", 10, "top k")
flags.DEFINE_list("target_item", [1679], "attacks target item")
flags.DEFINE_float("attack_size", 0.03, "attacks size")
flags.DEFINE_string("attack_type", "TNA", "attacks type")
flags.DEFINE_float("data_size", 0.4, "pass")
flags.DEFINE_float("meta_size", 1., "pass")
flags.DEFINE_integer('target_index', 0, 'select target items')
flags.DEFINE_integer('extend', 0, 'the number of ERM users')
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu


def get_rs(rs, dataset):
    rs_all = {"autorec": AutoRec, "svd": SVD, "bpr": BPR, 'cdae': CDAE, "pmf": PMF, "neumf": NeuMF, "slim": SLIM,
              "lrec": LRec, 'nmf': NMF, 'vaecf': VAE_CF, 'slopeone': Slopeone}
    rs = rs_all[rs](dataset)
    return rs


def train_rs(rs, dataset):
    RS = get_rs(rs, dataset)
    tf.reset_default_graph()
    RS.build_graph()
    print("Initialize %s" % rs)
    test_hr, test_ndcg = RS.train(dataset, FLAGS.is_train)
    hr, ndcg, ps, rank = utils.recommend(RS, dataset, FLAGS.target_item, FLAGS.top_k)
    return RS, hr


def norm(v):
    x = v / np.max(v)
    return x


if __name__ == '__main__':
    extend = 0

    # target items
    a = [[1485, 1320, 821, 1562, 1531],
         [1018, 946, 597, 575, 516],
         [3639, 3698, 3622, 3570, 3503],
         [1032, 3033, 2797, 2060, 1366],
         [1576, 926, 942, 848, 107],
         [539, 117, 1600, 1326, 208],
         [2504, 19779, 9624, 24064, 17390],
         [2417, 21817, 13064, 3348, 15085]]
    FLAGS.target_item = a[FLAGS.target_index]

    # initialize dataset
    dataset = Dataset(FLAGS.path + FLAGS.dataset, FLAGS.reg_data)
    dataset_temp = copy.deepcopy(dataset)
    attack_size = int(dataset.full_num_users * FLAGS.attack_size)
    filler_size = int(np.sum(dataset.trainMatrix.toarray() != 0) / dataset.num_users)
    # rating_size = np.sort(np.sum(dataset.trainMatrix.toarray() != 0, axis=1))
    # filler_size = np.median(rating_size)
    print("filler", filler_size)

    data_size = FLAGS.data_size

    # RS, hr = train_rs(FLAGS.rs, dataset)
    # print(hr)

    poison_user = utils.generate_mixfake(1, dataset, filler_size)
    poison_user = utils.random_mask(poison_user, 0)
    poison_user[:, FLAGS.target_item] = 1.
    poison_user1 = utils.generate_mixfake(attack_size, dataset, filler_size)

    repeat = 3
    if (FLAGS.dataset == 'filmtrust'):
        repeat = 5

    # attacks
    pre_poison_user = 0
    iters = 1
    if (FLAGS.is_train):
        for ii in range(iters):
            dataset1 = utils.estimate_dataset(copy.deepcopy(dataset), poison_user)
            influence_list = []
            RS, hr = train_rs(FLAGS.local_rs, dataset1)
            for it in range(repeat):
                influence = RS.influence_user(poison_user)
                influence_list.append(influence)

            for i in range(attack_size):
                influence = 0
                for cur_inf in influence_list:
                    influence += norm(cur_inf[0])
                influence[influence == 0] = np.min(influence)
                temp_candidates = utils.generate_mixfake2(100, dataset, filler_size)
                idx = np.argmax(
                    np.sum(temp_candidates * np.expand_dims(influence, axis=0), axis=1))
                poison_user1[i] = temp_candidates[idx]

            poison_user1[:, FLAGS.target_item] = 1.

        np.save("temp/%s/full/mixinf_poisoning_%d_%d_%f.npy" % (
            FLAGS.dataset, FLAGS.target_item[0], attack_size, FLAGS.data_size),
                poison_user1)
    poison_user = np.load("temp/%s/full/mixinf_poisoning_%d_%d_%f.npy" % (
        FLAGS.dataset, FLAGS.target_item[0], attack_size, FLAGS.data_size))

    # FLAGS.data_size = 1.
    # dataset = Dataset(FLAGS.path + FLAGS.dataset, FLAGS.reg_data)
    # all_hr = 0
    # for i in range(5):
    #     dataset1 = utils.estimate_dataset(copy.deepcopy(dataset), poison_user)
    #     RS, hr = train_rs(FLAGS.rs, dataset1)
    #     all_hr += hr
    # print(all_hr / 5)

    # poison_user = np.load("temp/%s/full/%s_poisoning_%d_%d_0.400000.npy" % (
    #     FLAGS.dataset, FLAGS.attack_type, a[FLAGS.target_index][0], attack_size))
    # # poison_user = np.load("temp/%s/full/%s_poisoning_%d_%d_0.400000.npy" % (
    # #     FLAGS.dataset, FLAGS.attack_type, a[FLAGS.target_index][0], attack_size))
    # dataset1 = utils.estimate_dataset(copy.deepcopy(dataset), poison_user)
    # RS, hr = train_rs(FLAGS.rs, dataset1)
    # print(hr)
