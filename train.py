from pytorch_metric_learning import losses
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import dataloader
import random
import model
import yaml
import os
import util
import classifier_embeddings

# load config
Control = yaml.safe_load(open("config/Control.yaml"))
setting = Control['settings']
dataset = Control['dataset']

# load dataset config
CONFIG = yaml.safe_load(
    open("config/" + Control['CONFIG_PATH'][setting][dataset]))
os.environ['CUDA_VISIBLE_DEVICES'] = str(CONFIG['gpus'])

if CONFIG['manualSeed'] is None:
    CONFIG['manualSeed'] = random.randint(1, 10000)
print("Random Seed: ", CONFIG['manualSeed'])
random.seed(CONFIG['manualSeed'])
torch.manual_seed(CONFIG['manualSeed'])

cudnn.benchmark = True

if torch.cuda.is_available() and not CONFIG['cuda']:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
#######################################################################################

# create model
Generator = model.Generator(CONFIG)
Discriminator = model.Critic(CONFIG)
Encoder = model.Encoder(CONFIG)

# inits data
input_res = torch.FloatTensor(CONFIG['batch_size'], CONFIG['resSize'])
input_att = torch.FloatTensor(CONFIG['batch_size'], CONFIG['attSize'])
noise_gen = torch.FloatTensor(CONFIG['batch_size'], CONFIG['nz'])
input_label = torch.LongTensor(CONFIG['batch_size'])

# if cuda
if CONFIG['cuda']:
    torch.cuda.manual_seed_all(CONFIG['manualSeed'])
    Generator.cuda()
    Discriminator.cuda()
    Encoder.cuda()
    input_res = input_res.cuda()
    noise_gen, input_att = noise_gen.cuda(), input_att.cuda()
    input_label = input_label.cuda()

    # if multi gpu
    if len(str(CONFIG['gpus']).split(',')) > 1:
        Generator = nn.DataParallel(Generator)
        Discriminator = nn.DataParallel(Discriminator)
        Encoder = nn.DataParallel(Encoder)

# load data
data = dataloader.DATA_LOADER(CONFIG)
print("# of training samples: ", data.ntrain)

# loss
circle_criterion = losses.CircleLoss(m=CONFIG['m'], gamma=CONFIG['gamma'])

# setup optimizer
optimizerD = optim.AdamW(itertools.chain(Discriminator.parameters(), Encoder.parameters()),  lr=CONFIG['lr'],
                         betas=(CONFIG['beta1'], 0.999))
optimizerG = optim.AdamW(
    Generator.parameters(), lr=CONFIG['lr'], betas=(CONFIG['beta1'], 0.999))


def safeModel():
    model_path = './models/' + CONFIG['dataset'] + '_' + setting
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # check if folder exit or not
    i = 2
    save_model_dir = model_path + "/train_1"
    while os.path.isdir(save_model_dir):
        save_model_dir = model_path + "/train_" + str(i)
        i += 1
    os.mkdir(save_model_dir)

    return save_model_dir


def sample():
    batch_feature, batch_label, batch_att = data.next_batch(
        CONFIG['batch_size'])
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)
    input_label.copy_(util.map_label(batch_label, data.seenclasses))


def calc_gradient_penalty(Discriminator, real_data, fake_data, input_att):
    alpha = torch.rand(CONFIG['batch_size'], 1)
    alpha = alpha.expand(real_data.size())
    if CONFIG['cuda']:
        alpha = alpha.cuda()
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    if CONFIG['cuda']:
        interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)
    disc_interpolates = Discriminator(interpolates, input_att)
    ones = torch.ones(disc_interpolates.size())
    if CONFIG['cuda']:
        ones = ones.cuda()
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1)
                        ** 2).mean() * CONFIG['lambda1']
    return gradient_penalty


def generate_syn_feature(Generator, classes, attribute, num):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass * num, CONFIG['resSize'])
    syn_label = torch.LongTensor(nclass * num)
    syn_att = torch.FloatTensor(num, CONFIG['attSize'])
    syn_noise = torch.FloatTensor(num, CONFIG['nz'])
    if CONFIG['cuda']:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()

    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.copy_(iclass_att.repeat(num, 1))
        syn_noise.normal_(0, 1)
        with torch.no_grad():
            output = Generator(syn_noise, syn_att)
        syn_feature.narrow(0, i * num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i * num, num).fill_(iclass)
    return syn_feature, syn_label


best_D = 1000
best_G = 0
best_W = 0
best_unseen = 0
best_seen = 0
best_h = 0
best_ep = 0
best_cls_ep = 0
if CONFIG['save_model']:
    save_model_dir = safeModel()
else:
    save_model_dir = ""

logger = util.get_logger(
    './log/' + str(CONFIG['dataset']) + '_' + setting + '_' + str(CONFIG['manualSeed']) + '_train_1.log')
logger.info('start training!')
logger.info('dataset:%s, syn_num:%d, gzsl:%s, seed:%d, batch_size:%d, nepoch:%d, lr:%f, batch_cls:%d, cls_nepoch:%d, cls_clr:%f, m:%f, gamma:%d' %
            (CONFIG['dataset'], CONFIG['syn_num'], CONFIG['gzsl'], CONFIG['manualSeed'], CONFIG['batch_size'], CONFIG['nepoch'], CONFIG['lr'], CONFIG['batch_cls'], CONFIG['cls_nepoch'], CONFIG['classifier_lr'], CONFIG['m'], CONFIG['gamma']))
for epoch in range(CONFIG['nepoch']):
    for i in range(0, data.ntrain, CONFIG['batch_size']):
        ############################
        # (1) Update D network
        ###########################
        for p in Discriminator.parameters():
            p.requires_grad = True
        for p in Encoder.parameters():
            p.requires_grad = True

        for iter_d in range(CONFIG['critic_iter']):
            sample()
            Discriminator.zero_grad()
            Encoder.zero_grad()

            # train with realG
            embed_real = Encoder(input_res)

            criticD_real = Discriminator(input_res, input_att)
            criticD_real = criticD_real.mean()

            # train with fakeG
            noise_gen.normal_(0, 1)
            fake = Generator(noise_gen, input_att)
            criticD_fake = Discriminator(fake.detach(), input_att)
            criticD_fake = criticD_fake.mean()

            # gradient penalty
            gradient_penalty = calc_gradient_penalty(
                Discriminator, input_res, fake.data, input_att)
            Wasserstein_D = criticD_real - criticD_fake

            # embedding
            embed_cat_real = torch.cat([embed_real, input_att], dim=1)
            cle_loss_real = circle_criterion(
                embed_cat_real, input_label)

            Loss_D = criticD_fake - criticD_real + gradient_penalty + cle_loss_real

            Loss_D.backward()
            optimizerD.step()
        ############################
        # (2) Update G network
        ###########################
        for p in Discriminator.parameters():
            p.requires_grad = False
        for p in Encoder.parameters():
            p.requires_grad = False

        Generator.zero_grad()
        noise_gen.normal_(0, 1)
        fake = Generator(noise_gen, input_att)

        criticG_fake = Discriminator(fake, input_att)
        criticG_fake = criticG_fake.mean()

        Loss_G = -criticG_fake

        Loss_G.backward()
        optimizerG.step()

    if (epoch + 1) % CONFIG['lr_decay_epoch'] == 0:
        for param_group in optimizerD.param_groups:
            param_group['lr'] = param_group['lr'] * CONFIG['lr_dec_rate']
        for param_group in optimizerG.param_groups:
            param_group['lr'] = param_group['lr'] * CONFIG['lr_dec_rate']

    print(
        '[%d/%d] Loss_D: %.4f Loss_G: %.4f, Wasserstein_dist: %.4f, cle_real: %.4f'
        % (epoch, CONFIG['nepoch']-1, Loss_D, Loss_G, Wasserstein_D, cle_loss_real))

    ######################################################
    # evaluate the model, set G to evaluation mode
    Generator.eval()
    Encoder.eval()
    for p in Generator.parameters():
        p.requires_grad = False
    for p in Encoder.parameters():
        p.requires_grad = False

    save = False
    syn_feature, syn_label = generate_syn_feature(
        Generator, data.unseenclasses, data.attribute, CONFIG['syn_num'])
    train_X = torch.cat((data.train_feature, syn_feature), 0)
    train_Y = torch.cat((data.train_label, syn_label), 0)

    cls = classifier_embeddings.CLASSIFIER(
        train_X, train_Y, Encoder, data, CONFIG, epoch, round(best_h, 2), save_model_dir)

    if cls.H > best_h:
        best_unseen = cls.acc_unseen
        best_seen = cls.acc_seen
        best_h = cls.H
        best_ep = epoch
        best_cls_ep = cls.epoch

        best_D = Loss_D
        best_G = Loss_G
        best_W = Wasserstein_D
        best_cle_real = cle_loss_real

        logger.info('epoch: %d Loss_D: %.4f Loss_G: %.4f, Wasserstein_dist: %.4f | cle_real: %.4f' % (
            best_ep, best_D, best_G, best_W, best_cle_real))
        logger.info('cls_ep=%4d, unseen=%.4f, seen=%.4f, h=%.4f' %
                    (best_cls_ep, best_unseen, best_seen, best_h))

        save = True

    print('cls_ep=%4d, unseen=%.4f, seen=%.4f, h=%.4f' %
            (cls.epoch, cls.acc_unseen, cls.acc_seen, cls.H))
    print('best_gan_ep=%4d, cls_ep=%4d, unseen=%.4f, seen=%.4f, h=%.4f' %
            (best_ep, best_cls_ep, best_unseen, best_seen, best_h))
    print('-'*30)

    if CONFIG['save_model'] and save:
        torch.save(Generator.state_dict(), '%s/Generator_%d.pt' %
                   (save_model_dir, best_ep))
        torch.save(Discriminator.state_dict(), '%s/Discriminator_%d.pt' %
                   (save_model_dir, best_ep))
        torch.save(Encoder.state_dict(), '%s/Encoder_%d.pt' %
                   (save_model_dir, best_ep))

    # reset G to training mode
    Generator.train()
    Encoder.train()
    for p in Generator.parameters():
        p.requires_grad = True
    for p in Encoder.parameters():
        p.requires_grad = True
    ######################################################


logger.info('finish training!')
