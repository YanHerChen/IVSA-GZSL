import torch
import numpy as np
import scipy.io as sio
from sklearn import preprocessing


class DATA_LOADER(object):
    def __init__(self, CONFIG):
        self.CONFIG = CONFIG
        self.load_data()

    def __len__(self):
        return len(self.data['train_seen']['resnet_features'])

    def next_batch(self, batch_size):
        idx = torch.randperm(self.ntrain)[0:batch_size]
        batch_feature = self.train_feature[idx]
        batch_label = self.train_label[idx]
        batch_att = self.attribute[batch_label]

        return batch_feature, batch_label, batch_att

    def load_data(self):
        matcontent = sio.loadmat(
            self.CONFIG['dataroot'] + "/" + self.CONFIG['dataset'] + "/" + self.CONFIG['image_embedding'] + ".mat")
        feature = matcontent['features'].T
        self.all_file = matcontent['image_files']
        label = matcontent['labels'].astype(int).squeeze() - 1
        matcontent = sio.loadmat(
            self.CONFIG['dataroot'] + "/" + self.CONFIG['dataset'] + "/" + self.CONFIG['class_embedding'] + "_splits.mat")

        # numpy array index starts from 0, matlab starts from 1
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1

        self.attribute = torch.from_numpy(matcontent['att'].T).float()

        self.train_image_file = self.all_file[trainval_loc]
        self.test_seen_image_file = self.all_file[test_seen_loc]
        self.test_unseen_image_file = self.all_file[test_unseen_loc]

        scaler = preprocessing.MinMaxScaler()

        _train_feature = scaler.fit_transform(feature[trainval_loc])
        _test_seen_feature = scaler.transform(feature[test_seen_loc])
        _test_unseen_feature = scaler.transform(
            feature[test_unseen_loc])

        self.train_feature = torch.from_numpy(_train_feature).float()
        mx = self.train_feature.max()
        self.train_feature.mul_(1 / mx)
        self.train_label = torch.from_numpy(label[trainval_loc]).long()

        self.test_unseen_feature = torch.from_numpy(
            _test_unseen_feature).float()
        self.test_unseen_feature.mul_(1 / mx)
        self.test_unseen_label = torch.from_numpy(
            label[test_unseen_loc]).long()

        self.test_seen_feature = torch.from_numpy(
            _test_seen_feature).float()
        self.test_seen_feature.mul_(1 / mx)
        self.test_seen_label = torch.from_numpy(
            label[test_seen_loc]).long()

        self.seenclasses = torch.from_numpy(
            np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(
            np.unique(self.test_unseen_label.numpy()))

        self.ntrain = self.train_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(
            0, self.ntrain_class + self.ntest_class).long()

        self.attribute_seen = self.attribute[self.seenclasses]
