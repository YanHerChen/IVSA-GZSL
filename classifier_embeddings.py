import torch
import torch.nn as nn
import torch.optim as optim
import util
import model


class CLASSIFIER:
    # train_Y is interger
    def __init__(self, _train_X, _train_Y, encoder, data, CONFIG, train_epoch, best_accuracy, save_model_dir):
        self.train_epoch = train_epoch
        self.best_accuracy = best_accuracy
        self.save_model_dir = save_model_dir
        self.save_model = CONFIG['save_model']
        
        # data
        self.train_X = _train_X
        self.train_Y = _train_Y
        self.test_seen_feature = data.test_seen_feature
        self.test_seen_label = data.test_seen_label
        self.test_unseen_feature = data.test_unseen_feature
        self.test_unseen_label = data.test_unseen_label
        self.seenclasses = data.seenclasses
        self.unseenclasses = data.unseenclasses

        # parameter
        if CONFIG['gzsl']:
            self.nclass = CONFIG['nclass_all']
        else:
            self.nclass = data.unseenclasses.size(0)

        self.batch_size = CONFIG['batch_cls']
        self.input_dim = CONFIG['embedSize']
        self.cuda = CONFIG['cuda']
        self.nepoch = CONFIG['cls_nepoch']
        self.lr = CONFIG['classifier_lr']
        self.beta1 = CONFIG['beta1']

        # model
        self.Encoder = encoder
        
        self.clsNet = model.LINEAR_Classifier(self.input_dim, self.nclass)
        self.clsNet.apply(util.weights_init)
        self.criterion = nn.NLLLoss()

        # init
        self.input = torch.FloatTensor(self.batch_size, _train_X.size(1))
        self.label = torch.LongTensor(self.batch_size)

        # setup optimizer
        self.optimizer = optim.AdamW(
            self.clsNet.parameters(), lr=self.lr, betas=(0.5, 0.999))

        if self.cuda:
            self.clsNet.cuda()
            self.criterion.cuda()
            self.input = self.input.cuda()
            self.label = self.label.cuda()

        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.ntrain = self.train_X.size()[0]

        # gzsl or zsl
        if CONFIG['gzsl']:
            self.acc_seen, self.acc_unseen, self.H, self.epoch = self.gzsl_cls()
        else:
            self.acc, self.epoch = self.zsl_cls()

            
    def zsl_cls(self):
        best_acc = 0
        best_epoch = 0
        mean_loss = 0
            
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):
                self.clsNet.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size)
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)

                with torch.no_grad():
                    embed = self.Encoder(self.input)
                output = self.clsNet(embed)

                loss = self.criterion(output, self.label)
                mean_loss += loss.data
                loss.backward()
                self.optimizer.step()
            
            with torch.no_grad():
                acc = util.val(self.test_unseen_feature,
                            self.test_unseen_label, self.unseenclasses, self.batch_size, self.clsNet, self.Encoder, self.cuda)
                if acc > best_acc:
                    best_acc = acc
                    best_epoch = epoch
                    
                    if best_acc > self.best_accuracy and self.save_model:
                        torch.save(self.clsNet.state_dict(), '%s/clsNet_%d.pt' %
                                (self.save_model_dir, self.train_epoch))
                    
        return best_acc, best_epoch

    def gzsl_cls(self):
        best_H = 0
        best_seen = 0
        best_unseen = 0
        best_epoch = 0
        
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):
                self.clsNet.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size)
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)

                with torch.no_grad():
                    embed = self.Encoder(self.input)
                output = self.clsNet(embed)
                loss = self.criterion(output, self.label)

                loss.backward()
                self.optimizer.step()
            
            with torch.no_grad():
                acc_seen = util.val_gzsl(
                    self.test_seen_feature, self.test_seen_label, self.seenclasses, self.batch_size, self.clsNet, self.Encoder, self.cuda)
                acc_unseen = util.val_gzsl(
                    self.test_unseen_feature, self.test_unseen_label, self.unseenclasses, self.batch_size, self.clsNet, self.Encoder, self.cuda)
                if (acc_seen+acc_unseen) == 0:
                    print('a bug')
                    H = 0
                else:
                    H = 2*acc_seen*acc_unseen / (acc_seen+acc_unseen)
                    
                if H > best_H:
                    best_seen = acc_seen
                    best_unseen = acc_unseen
                    best_H = H
                    best_epoch = epoch
                    
                    if H > self.best_accuracy and self.save_model:
                        torch.save(self.clsNet.state_dict(), '%s/clsNet_%d.pt' %
                                (self.save_model_dir, self.train_epoch))
                    
        return best_seen, best_unseen, best_H, best_epoch

    def next_batch(self, batch_size):
        start = self.index_in_epoch
        # shuffle the data at the first epoch
        if self.epochs_completed == 0 and start == 0:
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
        # the last batch
        if start + batch_size > self.ntrain:
            self.epochs_completed += 1
            rest_num_examples = self.ntrain - start
            if rest_num_examples > 0:
                X_rest_part = self.train_X[start:self.ntrain]
                Y_rest_part = self.train_Y[start:self.ntrain]
            # shuffle the data
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
            # start next epoch
            start = 0
            self.index_in_epoch = batch_size - rest_num_examples
            end = self.index_in_epoch
            X_new_part = self.train_X[start:end]
            Y_new_part = self.train_Y[start:end]
            #print(start, end)
            if rest_num_examples > 0:
                return torch.cat((X_rest_part, X_new_part), 0), torch.cat((Y_rest_part, Y_new_part), 0)
            else:
                return X_new_part, Y_new_part
        else:
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            #print(start, end)
            # from index start to index end-1
            return self.train_X[start:end], self.train_Y[start:end]
