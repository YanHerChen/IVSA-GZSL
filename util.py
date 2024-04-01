import logging
import torch
import os


def get_logger(filename, verbosity=1, name=None):
    while os.path.isfile(filename):
        filename = filename.replace('.log', '')
        temp = filename.split('_')
        filename = temp[0] + '_' + temp[1] + '_' + \
            temp[2] + '_' + temp[3] + '_' + str(int(temp[4])+1) + '.log'

    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )

    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label == classes[i]] = i

    return mapped_label


def val_gzsl(test_X, test_label, target_classes, batch_size, clsNet, encoder, cuda):
    start = 0
    ntest = test_X.size()[0]
    predicted_label = torch.LongTensor(test_label.size())
    for i in range(0, ntest, batch_size):
        end = min(ntest, start + batch_size)
        with torch.no_grad():
            if cuda:
                embed = encoder(test_X[start:end].cuda())
                output = clsNet(embed)
            else:
                embed = encoder(test_X[start:end])
                output = clsNet(embed)
        _, predicted_label[start:end] = torch.max(output, 1)
        start = end

    acc = compute_per_class_acc_gzsl(
        test_label, predicted_label, target_classes)
    return acc


def compute_per_class_acc_gzsl(test_label, predicted_label, target_classes):
    acc_per_class = 0
    for i in target_classes:
        idx = (test_label == i)
        acc_per_class += float(
            torch.sum(test_label[idx] == predicted_label[idx])) / float(torch.sum(idx))
    acc_per_class /= target_classes.size(0)
    return acc_per_class


def val(test_X, test_label, target_classes, batch_size, clsNet, encoder, cuda):
    start = 0
    ntest = test_X.size()[0]
    predicted_label = torch.LongTensor(test_label.size())
    for i in range(0, ntest, batch_size):
        end = min(ntest, start+batch_size)
        with torch.no_grad():
            if cuda:
                embed = encoder(test_X[start:end].cuda())
                output = clsNet(embed)
            else:
                embed = encoder(test_X[start:end])
                output = clsNet(embed)
        _, predicted_label[start:end] = torch.max(output, 1)
        start = end

    acc = compute_per_class_acc(map_label(
        test_label, target_classes), predicted_label, target_classes.size(0))
    return acc


def compute_per_class_acc(test_label, predicted_label, nclass):
    acc_per_class = torch.FloatTensor(nclass).fill_(0)
    for i in range(nclass):
        idx = (test_label == i)
        acc_per_class[i] = float(
            torch.sum(test_label[idx] == predicted_label[idx])) / float(torch.sum(idx))
    return acc_per_class.mean()
