import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import dataloader
import model
import yaml
import os
import util
import random

# load config
Control = yaml.safe_load(open("config/Control.yaml"))
setting = Control['settings']
dataset = Control['dataset']

CONFIG = yaml.safe_load(
    open("config/train_config_" + setting + "_" + dataset + ".yaml"))
os.environ['CUDA_VISIBLE_DEVICES'] = str(CONFIG['gpus'])

cudnn.benchmark = True

#######################################################################################
# load data
data = dataloader.DATA_LOADER(CONFIG)
test_seen_feature = data.test_seen_feature
test_seen_label = data.test_seen_label
test_unseen_feature = data.test_unseen_feature
test_unseen_label = data.test_unseen_label
seenclasses = data.seenclasses
unseenclasses = data.unseenclasses

# init model
Encoder = model.Encoder(CONFIG)
clsNet = model.LINEAR_Classifier(CONFIG['embedSize'], CONFIG['nclass_all'])

# load pretrained model
Encoder_path = "pretrained/" + dataset + "/" + setting + "/Encoder_" + setting + "_" + dataset + ".pt"
clsNet_path = "pretrained/" + dataset + "/" + setting + "/clsNet_" + setting + "_" + dataset + ".pt"

# multi gpu or not
if len(str(CONFIG['gpus']).split(',')) > 1:
    Encoder = nn.DataParallel(Encoder)
    clsNet = nn.DataParallel(clsNet)

    Encoder.load_state_dict(
        {k.replace('module.', ''): v for k, v in torch.load(Encoder_path).items()})
    clsNet.load_state_dict(
        {k.replace('module.', ''): v for k, v in torch.load(clsNet_path).items()})
else:
    Encoder.load_state_dict(torch.load(Encoder_path))
    clsNet.load_state_dict(torch.load(clsNet_path))

# cuda
if CONFIG['cuda']:
    Encoder.cuda()
    clsNet.cuda()

# evaluate
Encoder.eval()
clsNet.eval()

for p in Encoder.parameters():
    p.requires_grad = False
for p in clsNet.parameters():
    p.requires_grad = False


#################################################################
acc_seen = util.val_gzsl(
    test_seen_feature, test_seen_label, seenclasses, CONFIG['batch_cls'], clsNet, Encoder, CONFIG['cuda'])
acc_unseen = util.val_gzsl(
    test_unseen_feature, test_unseen_label, unseenclasses, CONFIG['batch_cls'], clsNet, Encoder, CONFIG['cuda'])

H = H = 2*acc_seen*acc_unseen / (acc_seen+acc_unseen)


print('dataset: %s, setting: %s, unseen:%.4f, seen:%.4f, h:%.4f' % (dataset, setting, acc_unseen, acc_seen, H))
