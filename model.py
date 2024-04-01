import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Encoder(nn.Module):
    def __init__(self, CONFIG):
        super(Encoder, self).__init__()

        self.fc1 = nn.Linear(CONFIG['resSize'], CONFIG['embedSize'])
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self, features):
        embedding = self.relu(self.fc1(features))
        return embedding


class Generator(nn.Module):
    def __init__(self, CONFIG):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(CONFIG['attSize'] + CONFIG['nz'], CONFIG['ngh'])
        self.fc2 = nn.Linear(CONFIG['ngh'], CONFIG['resSize'])
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)

        self.apply(weights_init)

    def forward(self, noise, att):
        h = torch.cat((noise, att), 1)
        h = self.lrelu(self.fc1(h))
        h = self.relu(self.fc2(h))
        return h


class Critic(nn.Module):
    def __init__(self, CONFIG):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(CONFIG['resSize'] +
                             CONFIG['attSize'], CONFIG['ndh'])
        self.fc2 = nn.Linear(CONFIG['ndh'], 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

        self.apply(weights_init)

    def forward(self, x, att):
        h = torch.cat((x, att), 1)
        h = self.lrelu(self.fc1(h))
        h = self.fc2(h)
        return h


class Softmax_Classifier(nn.Module):
    def __init__(self, CONFIG):
        super(Softmax_Classifier, self).__init__()
        self.fc = nn.Linear(CONFIG['embedSize'] + CONFIG['attSize']
                            , CONFIG['nclass_seen'])
        self.activate = nn.Softmax(dim=1)

    def forward(self, embedding):
        pred = self.activate(self.fc(embedding))
        return pred


class LINEAR_Classifier(nn.Module):
    def __init__(self, embed_size, nclass):
        super(LINEAR_Classifier, self).__init__()
        self.fc = nn.Linear(embed_size, nclass)
        self.logic = nn.LogSoftmax(dim=1)

    def forward(self, x):
        o = self.logic(self.fc(x))
        return o
