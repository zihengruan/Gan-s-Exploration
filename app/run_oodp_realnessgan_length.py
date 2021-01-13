# coding: utf-8
import argparse
import os
import pickle

import numpy as np
import pandas as pd
import json
import torch
import tqdm
from sklearn.manifold import TSNE
from torch.utils.data.dataloader import DataLoader
from transformers import BertModel
from transformers.optimization import AdamW
from importlib import import_module
from sklearn.metrics import roc_auc_score

import metrics
from config import Config
from data_utils import OOSDataset, SMPDataset
from logger import Logger
from metrics import plot_confusion_matrix
from processor.oos_processor_v3 import OOSProcessor
from processor.smp_processor import SMPProcessor
from processor.smp_processor_v2 import SMPProcessor_v2
from processor.smp_processor_v3 import SMPProcessor_v3
from utils import check_manual_seed, save_realness_gan_model, load_realness_gan_model, save_model, load_model, output_cases, EarlyStopping
from utils import convert_to_int_by_threshold
from utils.visualization import scatter_plot, my_plot_roc, plot_train_test
from utils.tool import ErrorRateAt95Recall, save_result, save_feature, std_mean
from utils.loss import CategoricalLoss


SEED = 123
freeze_data = dict()
best_dev = -1
gross_result = {}

if torch.cuda.is_available():
    device = 'cuda'
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
else:
    device = 'cpu'
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor


def check_args(args):
    """to check if the args is ok"""
    if not (args.do_train or args.do_eval or args.do_test):
        raise argparse.ArgumentError('You should pass at least one argument for --do_train or --do_eval or --do_test')


def main(args):
    logger.info('Checking...')
    print('torch.cuda.is_available:', torch.cuda.is_available())
    # print('torch.cuda.current_device:', torch.cuda.current_device())
    logger.info('device: {}'.format(device))
    logger.info('ood: {}'.format(args.ood))
    SEED = args.seed
    gross_result['seed'] = args.seed
    logger.info('seed: {}'.format(SEED))
    logger.info('model: {}'.format(args.model))
    check_manual_seed(SEED)
    check_args(args)
    if 0 <= args.beta <= 1:
        logger.info('beta: {}'.format(args.beta))
    logger.info('mode: {}'.format(args.mode))
    logger.info('maxlen: {}'.format(args.maxlen))
    logger.info('minlen: {}'.format(args.minlen))
    logger.info('optim_mode: {}'.format(args.optim_mode))
    logger.info('length_weight: {}'.format(args.length_weight))
    logger.info('sample_weight: {}'.format(args.sample_weight))

    logger.info('Loading config...')
    bert_config = Config('config/bert.ini')
    bert_config = bert_config(args.bert_type)

    # for oos-eval dataset
    data_config = Config('config/data.ini')
    data_config = data_config(args.dataset)

    # Prepare data processor
    data_path = os.path.join(data_config['DataDir'], data_config[args.data_file])  # 把目录和文件名合成一个路径
    label_path = data_path.replace('.json', '.label')

    if args.dataset == 'oos-eval':
        processor = OOSProcessor(bert_config, maxlen=32)
    elif args.dataset == 'smp':
        if args.mode == -1:
            processor = SMPProcessor(bert_config, maxlen=32)
            print('processor')
        else:
            processor = SMPProcessor_v3(bert_config, maxlen=32)
            print('processor_v3')
    else:
        raise ValueError('The dataset {} is not supported.'.format(args.dataset))

    processor.load_label(label_path)  # Adding label_to_id and id_to_label ot processor.

    n_class = len(processor.id_to_label)
    print('label: ', processor.id_to_label)
    config = vars(args)  # 返回参数字典
    config['gan_save_path'] = os.path.join(args.output_dir, 'save', 'gan.pt')
    config['bert_save_path'] = os.path.join(args.output_dir, 'save', 'bert.pt')
    config['n_class'] = n_class

    logger.info('config:')
    logger.info(config)

    realness_model = import_module('model.' + args.model)
    vanilla_model = import_module('model.gan')

    realness_D = realness_model.Discriminator(config)
    classification_D = vanilla_model.Discriminator(config)
    G = realness_model.Generator(config)
    E = BertModel.from_pretrained(bert_config['PreTrainModelDir'])  # Bert encoder

    if args.fine_tune:
        for param in E.parameters():
            param.requires_grad = True
    else:
        for param in E.parameters():
            param.requires_grad = False

    realness_D.to(device)
    classification_D.to(device)
    G.to(device)
    E.to(device)

    global_step = 0

    def train(train_dataset, dev_dataset):
        train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=2)

        global best_dev
        nonlocal global_step
        n_sample = len(train_dataloader)
        early_stopping = EarlyStopping(args.patience, logger=logger)
        # Loss function
        adversarial_loss = torch.nn.BCELoss().to(device)
        classified_loss = torch.nn.CrossEntropyLoss().to(device)

        triplet_loss = CategoricalLoss(atoms=args.num_outcomes, v_max=args.positive_skew, v_min=args.negative_skew)
        triplet_loss.to(device)
        num_outcomes = args.num_outcomes

        # Optimizers
        # optimizer_G = torch.optim.Adam(G.parameters(), lr=args.G_lr)  # optimizer for generator
        # optimizer_D = torch.optim.Adam(D.parameters(), lr=args.D_lr)  # optimizer for discriminator
        optimizer_E = AdamW(E.parameters(), args.bert_lr)

        optimizer_G = torch.optim.Adam(G.parameters(), lr=args.G_lr, betas=(args.beta1, args.beta2),weight_decay=args.weight_decay, eps=args.adam_eps)
        D_parameters = list(realness_D.parameters()) + list(classification_D.parameters())
        optimizer_D = torch.optim.Adam(D_parameters, lr=args.D_lr, betas=(args.beta1, args.beta2),weight_decay=args.weight_decay)
        decayG = torch.optim.lr_scheduler.ExponentialLR(optimizer_G, gamma=1 - args.decay)
        decayD = torch.optim.lr_scheduler.ExponentialLR(optimizer_D, gamma=1 - args.decay)

        G_total_train_loss = []
        D_total_fake_loss = []
        D_total_real_loss = []
        FM_total_train_loss = []
        D_total_class_loss = []
        valid_detection_loss = []
        valid_oos_ind_precision = []
        valid_oos_ind_recall = []
        valid_oos_ind_f_score = []

        all_features = []
        result = dict()

        for i in range(args.n_epoch):

            # Initialize model state
            G.train()
            realness_D.train()
            classification_D.train()
            E.train()

            G_train_loss = 0
            G_d_loss = 0
            D_fake_loss = 0
            D_real_loss = 0
            FM_train_loss = 0
            D_class_loss = 0

            G_features = []

            # define anchors
            # e.g. normal and uniform
            gauss = np.random.normal(0, 0.1, 1000)
            count, bins = np.histogram(gauss, args.num_outcomes)
            anchor0 = count / sum(count)

            unif = np.random.uniform(-1, 1, 1000)
            count, bins = np.histogram(unif, args.num_outcomes)
            anchor1 = count / sum(count)

            for sample in tqdm.tqdm(train_dataloader):
                sample = (i.to(device) for i in sample)
                if args.dataset == 'smp':
                    token, mask, type_ids, knowledge_tag, y = sample
                    batch = len(token)

                    ood_sample = (y == 0.0).float()
                    # weight = torch.ones(len(ood_sample)).to(device) - ood_sample * args.beta
                    # real_loss_func = torch.nn.BCELoss(weight=weight).to(device)

                    # length weight
                    length_sample = FloatTensor([0] * batch)
                    if args.minlen != -1:
                        short_sample = (mask[:, args.minlen] == 0).float()
                        length_sample = length_sample.add(short_sample)
                    if args.maxlen != -1:
                        long_sample = mask[:, args.maxlen].float()
                        length_sample = length_sample.add(long_sample)

                    # get knowledge sample weight by knowledge_tag
                    exclude_sample = knowledge_tag

                    # initailize weight
                    weight = torch.ones(batch).to(device)

                    # optimize without weights
                    if args.optim_mode == 0 and 0 <= args.beta <= 1:
                        weight -= ood_sample * args.beta

                    # only optimize length by weight
                    if args.optim_mode == 1:
                        # set all exclude_sample's weight to 0
                        weight -= exclude_sample
                        length_sample -= exclude_sample
                        length_sample = (length_sample > 0).float()
                        weight -= length_sample * (1 - args.length_weight)

                        # set ood sample weight
                        if 0 <= args.beta <= 1:
                            ood_sample -= exclude_sample
                            ood_sample = (ood_sample > 0).float()
                            temp = torch.ones(batch).to(device)
                            temp -= ood_sample * args.beta
                            weight *= temp

                    # only optimize sample by weight
                    if args.optim_mode == 2:
                        # set all length_sample's weight to 0
                        weight -= length_sample

                        exclude_sample -= length_sample
                        exclude_sample = (exclude_sample > 0).float()
                        weight -= exclude_sample * (1 - args.sample_weight)

                        # set ood sample weight
                        if 0 <= args.beta <= 1:
                            ood_sample -= length_sample
                            ood_sample = (ood_sample > 0).float()
                            temp = torch.ones(batch)
                            temp -= ood_sample * args.beta
                            weight *= temp

                    # optimize length and sample by weight
                    # if args.optim_mode == 3:
                    #     alpha = 0.5
                    #     beta = 0.5
                    #     weight = torch.ones(len(length_sample)).to(device) \
                    #              - alpha * length_sample * (1 - args.length_weight) \
                    #              - beta * exclude_sample * (1 - args.sample_weight)

                if args.dataset == 'oos-eval':
                    token, mask, type_ids, y = sample
                    batch = len(token)

                    ood_sample = (y == 0.0).float()
                    # weight = torch.ones(len(ood_sample)).to(device) - ood_sample * args.beta
                    # real_loss_func = torch.nn.BCELoss(weight=weight).to(device)

                    # length weight
                    length_sample = FloatTensor([0] * batch)
                    if args.minlen != -1:
                        short_sample = (mask[:, args.minlen] == 0).float()
                        length_sample = length_sample.add(short_sample)
                    if args.maxlen != -1:
                        long_sample = mask[:, args.maxlen].float()
                        length_sample = length_sample.add(long_sample)

                    # initailize weight
                    weight = torch.ones(batch).to(device)

                    # optimize without weights
                    if args.optim_mode == 0 and 0 <= args.beta <= 1:
                        weight -= ood_sample * args.beta

                    # only optimize length by weight
                    if args.optim_mode == 1:
                        weight -= length_sample * (1 - args.length_weight)

                        # set ood sample weight
                        if 0 <= args.beta <= 1:
                            ood_sample -= length_sample
                            ood_sample = (ood_sample > 0).float()
                            temp = torch.ones(batch).to(device)
                            temp -= ood_sample * args.beta
                            weight *= temp

                real_loss_func = torch.nn.BCELoss(weight=weight).to(device)

                # the label used to train generator and discriminator.
                valid_label = FloatTensor(batch, 1).fill_(1.0).detach()
                fake_label = FloatTensor(batch, 1).fill_(0.0).detach()

                anchor_real = torch.zeros((batch, num_outcomes), dtype=torch.float).to(device) + torch.tensor(anchor1, dtype=torch.float).to(device)
                anchor_fake = torch.zeros((batch, num_outcomes), dtype=torch.float).to(device) + torch.tensor(anchor0, dtype=torch.float).to(device)

                optimizer_E.zero_grad()
                sequence_output, pooled_output = E(token, mask, type_ids)
                real_feature = pooled_output

                for t in range(args.D_updates):
                    # train D on real
                    optimizer_D.zero_grad()
                    realness_discriminator_output= realness_D(real_feature).log_softmax(1).exp()
                    realness_discriminator_output = realness_discriminator_output.squeeze()

                    realness_real_loss = triplet_loss(anchor_real, realness_discriminator_output, skewness=args.positive_skew)

                    real_f_vector, classification_discriminator_output, classification_output = classification_D(real_feature, return_feature=True)
                    classification_discriminator_output = classification_discriminator_output.squeeze()

                    # real_loss = adversarial_loss(classification_discriminator_output, (y != 0.0).float())
                    classification_real_loss = real_loss_func(classification_discriminator_output, (y != 0.0).float())

                    real_loss = realness_real_loss + classification_real_loss
                    if n_class > 2:  # 大于2表示除了训练判别器还要训练分类器
                        class_loss = classified_loss(classification_output, y.long())
                        real_loss += class_loss
                        D_class_loss += class_loss.detach()
                    real_loss.backward()

                    if args.do_vis:
                        all_features.append(real_f_vector.detach())

                    # # train D on fake
                    if args.model == 'lstm_gan' or args.model == 'cnn_gan':
                        z = FloatTensor(np.random.normal(0, 1, (batch, 32, args.G_z_dim))).to(device)
                    else:
                        # uniform (-1,1)
                        # z = FloatTensor(np.random.uniform(-1, 1, (batch, args.G_z_dim))).to(device)
                        z = FloatTensor(np.random.normal(0, 1, (batch, args.G_z_dim))).to(device)
                    fake_feature = G(z).detach()

                    realness_fake_discriminator_output= realness_D(fake_feature).log_softmax(1).exp()
                    classification_fake_discriminator_output = classification_D.detect_only(fake_feature)

                    realness_fake_loss = triplet_loss(anchor_fake, realness_fake_discriminator_output, skewness=args.negative_skew)

                    # beta of fake
                    if 0 <= args.beta <= 1:
                        classification_fake_loss = args.beta * adversarial_loss(classification_fake_discriminator_output, fake_label)
                    else:
                        classification_fake_loss = adversarial_loss(classification_fake_discriminator_output, fake_label)

                    fake_loss = realness_fake_loss + classification_fake_loss
                    fake_loss.backward()
                    optimizer_D.step()
                decayD.step()

                if args.fine_tune:
                    optimizer_E.step()

                for t in range(args.G_updates):
                    # train G
                    optimizer_G.zero_grad()

                    sequence_output, pooled_output = E(token, mask, type_ids)
                    real_feature = pooled_output
                    realness_discriminator_output = realness_D(real_feature).log_softmax(1).exp()
                    realness_discriminator_output = realness_discriminator_output.squeeze()

                    if args.model == 'lstm_gan' or args.model == 'cnn_gan':
                        z = FloatTensor(np.random.normal(0, 1, (batch, 32, args.G_z_dim))).to(device)
                    else:
                        # uniform (-1,1)
                        # z = FloatTensor(np.random.uniform(-1, 1, (batch, args.G_z_dim))).to(device)
                        z = FloatTensor(np.random.normal(0, 1, (batch, args.G_z_dim))).to(device)
                    realness_D_decision = realness_D(G(z)).log_softmax(1).exp()

                    fake_f_vector, classification_D_decision = classification_D.detect_only(G(z), return_feature=True)

                    if args.do_vis:
                        G_features.append(fake_f_vector.detach())

                    classification_gd_loss = adversarial_loss(classification_D_decision, valid_label)
                    # feature matching loss
                    fm_loss = torch.abs(torch.mean(real_f_vector.detach(), 0) - torch.mean(fake_f_vector, 0)).mean()
                    # fm_loss = feature_matching_loss(torch.mean(fake_f_vector, 0), torch.mean(real_f_vector.detach(), 0))
                    classification_g_loss = classification_gd_loss + 0 * fm_loss

                    if args.relativisticG:
                        realness_g_loss = -triplet_loss(anchor_fake, realness_D_decision, skewness=args.negative_skew) + triplet_loss(realness_discriminator_output, realness_D_decision)
                    else:
                        realness_g_loss = -triplet_loss(anchor_fake, realness_D_decision, skewness=args.negative_skew) + triplet_loss(anchor_real, realness_D_decision, skewness=args.positive_skew)

                    g_loss = realness_g_loss + classification_gd_loss
                    g_loss.backward()
                    optimizer_G.step()
                decayG.step()

                global_step += 1

                D_fake_loss += fake_loss.detach()
                D_real_loss += real_loss.detach()
                # G_d_loss += g_loss.detach()
                G_train_loss += g_loss.detach()# + fm_loss.detach()
                # FM_train_loss += fm_loss.detach()

                # logger.info('[Epoch {}] Train: D_fake_loss: {}'.format(i, D_fake_loss / n_sample))
                # logger.info('[Epoch {}] Train: D_real_loss: {}'.format(i, D_real_loss / n_sample))
                # logger.info('[Epoch {}] Train: D_class_loss: {}'.format(i, D_class_loss / n_sample))
                # logger.info('[Epoch {}] Train: G_train_loss: {}'.format(i, G_train_loss / n_sample))
                # logger.info('[Epoch {}] Train: G_d_loss: {}'.format(i, G_d_loss / n_sample))
                # logger.info('[Epoch {}] Train: FM_train_loss: {}'.format(i, FM_train_loss / n_sample))
                # logger.info('---------------------------------------------------------------------------')

            D_total_fake_loss.append(D_fake_loss / n_sample)
            D_total_real_loss.append(D_real_loss / n_sample)
            # D_total_class_loss.append(D_class_loss / n_sample)
            G_total_train_loss.append(G_train_loss / n_sample)
            # FM_total_train_loss.append(FM_train_loss / n_sample)

            if dev_dataset:
                # logger.info('#################### eval result at step {} ####################'.format(global_step))
                eval_result = eval(dev_dataset)

                if args.do_vis and args.do_g_eval_vis:
                    G_features = torch.cat(G_features, 0).cpu().numpy()

                    features = np.concatenate([eval_result['all_features'], G_features], axis=0)
                    features = TSNE(n_components=2, verbose=1, n_jobs=-1).fit_transform(features)
                    labels = np.concatenate([eval_result['all_binary_y'], np.array([-1] * len(G_features))], 0).reshape(
                        -1, 1)

                    data = np.concatenate([features, labels], 1)
                    fig = scatter_plot(data, processor)
                    fig.savefig(os.path.join(args.output_dir, 'plot_epoch_' + str(i) + '.png'))

                valid_detection_loss.append(eval_result['detection_loss'])
                valid_oos_ind_precision.append(eval_result['oos_ind_precision'])
                valid_oos_ind_recall.append(eval_result['oos_ind_recall'])
                valid_oos_ind_f_score.append(eval_result['oos_ind_f_score'])

                # 1 表示要保存模型
                # 0 表示不需要保存模型
                # -1 表示不需要模型，且超过了patience，需要early stop
                signal = early_stopping(-eval_result['eer'])
                if signal == -1:
                    break
                elif signal == 0:
                    pass
                elif signal == 1:
                    save_realness_gan_model(realness_D, classification_D, G, config['gan_save_path'])
                    if args.fine_tune:
                        save_model(E, path=config['bert_save_path'], model_name='bert')

                # logger.info(eval_result)
                # logger.info('valid_eer: {}'.format(eval_result['eer']))
                # logger.info('valid_oos_ind_precision: {}'.format(eval_result['oos_ind_precision']))
                # logger.info('valid_oos_ind_recall: {}'.format(eval_result['oos_ind_recall']))
                # logger.info('valid_oos_ind_f_score: {}'.format(eval_result['oos_ind_f_score']))
                # logger.info('valid_auc: {}'.format(eval_result['auc']))
                # logger.info(
                #     'valid_fpr95: {}'.format(ErrorRateAt95Recall(eval_result['all_binary_y'], eval_result['y_score'])))

        if args.patience >= args.n_epoch:
            save_realness_gan_model(realness_D, classification_D, G, config['gan_save_path'])
            if args.fine_tune:
                save_model(E, path=config['bert_save_path'], model_name='bert')

        freeze_data['D_total_fake_loss'] = D_total_fake_loss
        freeze_data['D_total_real_loss'] = D_total_real_loss
        freeze_data['D_total_class_loss'] = D_total_class_loss
        freeze_data['G_total_train_loss'] = G_total_train_loss
        freeze_data['FM_total_train_loss'] = FM_total_train_loss
        freeze_data['valid_real_loss'] = valid_detection_loss
        freeze_data['valid_oos_ind_precision'] = valid_oos_ind_precision
        freeze_data['valid_oos_ind_recall'] = valid_oos_ind_recall
        freeze_data['valid_oos_ind_f_score'] = valid_oos_ind_f_score

        best_dev = -early_stopping.best_score

        if args.do_vis:
            all_features = torch.cat(all_features, 0).cpu().numpy()
            result['all_features'] = all_features
        return result

    def eval(dataset):
        dev_dataloader = DataLoader(dataset, batch_size=args.predict_batch_size, shuffle=False, num_workers=2)
        n_sample = len(dev_dataloader)
        result = dict()

        # Loss function
        detection_loss = torch.nn.BCELoss().to(device)
        classified_loss = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)
        triplet_loss = CategoricalLoss(atoms=args.num_outcomes, v_max=args.positive_skew, v_min=args.negative_skew)
        triplet_loss.to(device)

        # define anchors
        # e.g. normal and uniform
        gauss = np.random.normal(0, 0.1, 1000)
        count, bins = np.histogram(gauss, args.num_outcomes)
        anchor0 = count / sum(count)

        unif = np.random.uniform(-1, 1, 1000)
        count, bins = np.histogram(unif, args.num_outcomes)
        anchor1 = count / sum(count)

        G.eval()
        realness_D.eval()
        classification_D.eval()
        E.eval()

        all_detection_preds = []
        all_class_preds = []
        all_features = []

        for sample in tqdm.tqdm(dev_dataloader):
            sample = (i.to(device) for i in sample)
            if args.dataset == 'smp':
                token, mask, type_ids, knowledge_tag, y = sample
            if args.dataset == 'oos-eval':
                token, mask, type_ids, y = sample
            batch = len(token)

            # -------------------------evaluate D------------------------- #
            # BERT encode sentence to feature vector

            num_outcomes = args.num_outcomes
            anchor_real = torch.zeros(num_outcomes, dtype=torch.float).to(device) + torch.tensor(anchor1, dtype=torch.float).to(device)
            anchor_fake = torch.zeros(num_outcomes, dtype=torch.float).to(device) + torch.tensor(anchor0, dtype=torch.float).to(device)

            with torch.no_grad():
                sequence_output, pooled_output = E(token, mask, type_ids)
                real_feature = pooled_output

                # 大于2表示除了训练判别器还要训练分类器
                if n_class > 2:
                    f_vector, classification_discriminator_output, classification_output = classification_D(real_feature)
                    all_detection_preds.append(classification_discriminator_output)
                    all_class_preds.append(classification_output)

                # 只预测判别器
                else:
                    realness_discriminator_output = realness_D(real_feature).log_softmax(1).exp()
                    f_vector, classification_discriminator_output = classification_D.detect_only(real_feature, return_feature=True)

                    # 用 classification_D 做分类的判别结果
                    all_detection_preds.append(classification_discriminator_output)

                if args.do_vis:
                    all_features.append(f_vector)

        #             divergence_to_preidction = []
        #             for output in realness_discriminator_output:
        #                 d_real = triplet_loss(anchor_real, output)
        #                 d_fake = triplet_loss(anchor_fake, output)
        #                 divergence_to_preidction.append(1 if d_real > d_fake else 0)
        #             all_detection_preds.extend(divergence_to_preidction)
        #
        # # 用 realness_D 做分类的判别结果
        # all_detection_preds = LongTensor(all_detection_preds).cpu()
        # all_detection_binary_preds = all_detection_preds  # [length, 1]


        all_y = LongTensor(dataset.dataset[:, -1].astype(int)).cpu()  # [length, n_class]
        all_binary_y = (all_y != 0).long()  # [length, 1] label 0 is oos
        all_detection_preds = torch.cat(all_detection_preds, 0).cpu()  # [length, 1]
        all_detection_binary_preds = convert_to_int_by_threshold(all_detection_preds.squeeze())  # [length, 1]

        # print('all_detection_preds', all_detection_preds.size())
        # print('all_binary_y', all_binary_y.size())
        # 计算损失
        # detection_loss = detection_loss(all_detection_preds.float(), all_binary_y.float()) # 用 realness_D 做分类的判别结果
        detection_loss = detection_loss(all_detection_preds.squeeze(), all_binary_y.float()) # 用 classification_D 做分类的判别结果
        result['detection_loss'] = detection_loss

        if n_class > 2:
            class_one_hot_preds = torch.cat(all_class_preds, 0).detach().cpu()  # one hot label
            class_loss = classified_loss(class_one_hot_preds, all_y)  # compute loss
            all_class_preds = torch.argmax(class_one_hot_preds, 1)  # label
            class_acc = metrics.ind_class_accuracy(all_class_preds, all_y, oos_index=0)  # accuracy for ind class
            logger.info(metrics.classification_report(all_y, all_class_preds, target_names=processor.id_to_label))

        # logger.info(metrics.classification_report(all_binary_y, all_detection_binary_preds, target_names=['oos', 'in']))

        # report
        oos_ind_precision, oos_ind_recall, oos_ind_fscore, _ = metrics.binary_recall_fscore(all_detection_binary_preds,
                                                                                            all_binary_y)
        detection_acc = metrics.accuracy(all_detection_binary_preds, all_binary_y)

        y_score = all_detection_preds.squeeze().tolist()
        eer = metrics.cal_eer(all_binary_y, y_score)

        result['eer'] = eer
        result['all_detection_binary_preds'] = all_detection_binary_preds
        result['detection_acc'] = detection_acc
        result['all_binary_y'] = all_binary_y
        result['oos_ind_precision'] = oos_ind_precision
        result['oos_ind_recall'] = oos_ind_recall
        result['oos_ind_f_score'] = oos_ind_fscore
        result['y_score'] = y_score
        result['auc'] = roc_auc_score(all_binary_y, y_score)
        result['fpr95'] = ErrorRateAt95Recall(all_binary_y, y_score)
        if n_class > 2:
            result['class_loss'] = class_loss
            result['class_acc'] = class_acc
        if args.do_vis:
            all_features = torch.cat(all_features, 0).cpu().numpy()
            result['all_features'] = all_features

        freeze_data['valid_all_y'] = all_y
        freeze_data['vaild_all_pred'] = all_detection_binary_preds
        freeze_data['valid_score'] = y_score

        return result

    def test(dataset):
        # load BERT and GAN
        load_realness_gan_model(realness_D, classification_D, G, config['gan_save_path'])
        if args.fine_tune:
            load_model(E, path=config['bert_save_path'], model_name='bert')

        test_dataloader = DataLoader(dataset, batch_size=args.predict_batch_size, shuffle=False, num_workers=2)
        n_sample = len(test_dataloader)
        result = dict()

        # Loss function
        detection_loss = torch.nn.BCELoss().to(device)
        classified_loss = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)

        G.eval()
        realness_D.eval()
        classification_D.eval()
        E.eval()

        all_detection_preds = []
        all_class_preds = []
        all_features = []

        for sample in tqdm.tqdm(test_dataloader):
            sample = (i.to(device) for i in sample)
            if args.dataset == 'smp':
                token, mask, type_ids, knowledge_tag, y = sample
            if args.dataset == 'oos-eval':
                token, mask, type_ids, y = sample
            batch = len(token)

            # -------------------------evaluate D------------------------- #
            # BERT encode sentence to feature vector

            with torch.no_grad():
                sequence_output, pooled_output = E(token, mask, type_ids)
                real_feature = pooled_output

                # 大于2表示除了训练判别器还要训练分类器
                if n_class > 2:
                    f_vector, classification_discriminator_output, classification_output = classification_D(real_feature, return_feature=True)
                    all_detection_preds.append(classification_discriminator_output)
                    all_class_preds.append(classification_output)

                # 只预测判别器
                else:
                    f_vector, classification_discriminator_output = classification_D.detect_only(real_feature, return_feature=True)
                    all_detection_preds.append(classification_discriminator_output)
                if args.do_vis:
                    all_features.append(f_vector)

        all_y = LongTensor(dataset.dataset[:, -1].astype(int)).cpu()  # [length, n_class]
        all_binary_y = (all_y != 0).long()  # [length, 1] label 0 is oos
        all_detection_preds = torch.cat(all_detection_preds, 0).cpu()  # [length, 1]
        all_detection_binary_preds = convert_to_int_by_threshold(all_detection_preds.squeeze())  # [length, 1]

        # 计算损失
        detection_loss = detection_loss(all_detection_preds, all_binary_y.float())
        result['detection_loss'] = detection_loss

        if n_class > 2:
            class_one_hot_preds = torch.cat(all_class_preds, 0).detach().cpu()  # one hot label
            class_loss = classified_loss(class_one_hot_preds, all_y)  # compute loss
            all_class_preds = torch.argmax(class_one_hot_preds, 1)  # label
            class_acc = metrics.ind_class_accuracy(all_class_preds, all_y, oos_index=0)  # accuracy for ind class
            logger.info(metrics.classification_report(all_y, all_class_preds, target_names=processor.id_to_label))

        # logger.info(metrics.classification_report(all_binary_y, all_detection_binary_preds, target_names=['oos', 'in']))

        # report
        oos_ind_precision, oos_ind_recall, oos_ind_fscore, _ = metrics.binary_recall_fscore(all_detection_binary_preds,
                                                                                            all_binary_y)
        detection_acc = metrics.accuracy(all_detection_binary_preds, all_binary_y)

        y_score = all_detection_preds.squeeze().tolist()
        eer = metrics.cal_eer(all_binary_y, y_score)

        result['eer'] = eer
        result['all_detection_binary_preds'] = all_detection_binary_preds
        result['detection_acc'] = detection_acc
        result['all_binary_y'] = all_binary_y
        result['all_y'] = all_y
        result['oos_ind_precision'] = oos_ind_precision
        result['oos_ind_recall'] = oos_ind_recall
        result['oos_ind_f_score'] = oos_ind_fscore
        result['score'] = y_score
        result['y_score'] = y_score
        result['auc'] = roc_auc_score(all_binary_y, y_score)
        result['fpr95'] = ErrorRateAt95Recall(all_binary_y, y_score)
        if n_class > 2:
            result['class_loss'] = class_loss
            result['class_acc'] = class_acc
        if args.do_vis:
            all_features = torch.cat(all_features, 0).cpu().numpy()
            result['all_features'] = all_features

        freeze_data['test_all_y'] = all_y.tolist()
        freeze_data['test_all_pred'] = all_detection_binary_preds.tolist()
        freeze_data['test_score'] = y_score

        return result

    def get_fake_feature(num_output):
        """
        生成一定数量的假特征
        """
        G.eval()
        fake_features = []
        start = 0
        batch = args.predict_batch_size
        with torch.no_grad():
            while start < num_output:
                end = min(num_output, start + batch)
                if args.model == 'lstm_gan' or args.model == 'cnn_gan':
                    z = FloatTensor(np.random.normal(0, 1, size=(end - start, 32, args.G_z_dim)))
                else:
                    z = FloatTensor(np.random.normal(0, 1, size=(end - start, args.G_z_dim)))
                fake_feature = G(z)
                f_vector, _ = classification_D.detect_only(fake_feature, return_feature=True)
                fake_features.append(f_vector)
                start += batch
            return torch.cat(fake_features, 0).cpu().numpy()

    if args.do_train:
        if config['data_file'].startswith('binary'):
            if args.optim_mode == 0:
                text_train_set = processor.read_dataset(data_path, ['train'], args.mode, args.maxlen, args.minlen,
                                                        pre_exclude=True)
            else:
                # optimize length or sample by weight
                text_train_set = processor.read_dataset(data_path, ['train'], args.mode, args.maxlen, args.minlen,
                                                        pre_exclude=False)

            text_dev_set = processor.read_dataset(data_path, ['val'], args.mode, args.maxlen, args.minlen,
                                                  pre_exclude=True)

        elif config['dataset'] == 'oos-eval':
            text_train_set = processor.read_dataset(data_path, ['train', 'oos_train'])
            text_dev_set = processor.read_dataset(data_path, ['val', 'oos_val'])
        elif config['dataset'] == 'smp':
            text_train_set, text_train_len = processor.read_dataset(data_path, ['train'])
            text_dev_set, text_dev_len = processor.read_dataset(data_path, ['val'])

        if args.ood:
            text_train_set = [sample for sample in text_train_set if sample['domain'] != 'chat']

        train_features = processor.convert_to_ids(text_train_set)
        dev_features = processor.convert_to_ids(text_dev_set)

        if config['dataset'] == 'oos-eval':
            train_dataset = OOSDataset(train_features)
            dev_dataset = OOSDataset(dev_features)
        if config['dataset'] == 'smp':
            train_dataset = SMPDataset(train_features)
            dev_dataset = SMPDataset(dev_features)

        train_result = train(train_dataset, dev_dataset)
        # save_feature(train_result['all_features'], os.path.join(args.output_dir, 'train_feature'))

    if args.do_eval:
        logger.info('#################### eval result at step {} ####################'.format(global_step))
        if config['data_file'].startswith('binary'):
            #  don't optim dev_set by weight, don't pre_exclude it
            text_dev_set = processor.read_dataset(data_path, ['val'], args.mode, args.maxlen, args.minlen,
                                                  pre_exclude=False)

        elif config['dataset'] == 'oos-eval':
            text_dev_set = processor.read_dataset(data_path, ['val', 'oos_val'])
        elif config['dataset'] == 'smp':
            text_dev_set = processor.read_dataset(data_path, ['val'])

        dev_features = processor.convert_to_ids(text_dev_set)

        if config['dataset'] == 'oos-eval':
            dev_dataset = OOSDataset(dev_features)
        if config['dataset'] == 'smp':
            dev_dataset = SMPDataset(dev_features)

        eval_result = eval(dev_dataset)
        # logger.info(eval_result)
        logger.info('eval_eer: {}'.format(eval_result['eer']))
        logger.info('eval_oos_ind_precision: {}'.format(eval_result['oos_ind_precision']))
        logger.info('eval_oos_ind_recall: {}'.format(eval_result['oos_ind_recall']))
        logger.info('eval_oos_ind_f_score: {}'.format(eval_result['oos_ind_f_score']))
        logger.info('eval_auc: {}'.format(eval_result['auc']))
        logger.info(
            'eval_fpr95: {}'.format(ErrorRateAt95Recall(eval_result['all_binary_y'], eval_result['y_score'])))
        gross_result['eval_oos_ind_precision'] = eval_result['oos_ind_precision']
        gross_result['eval_oos_ind_recall'] = eval_result['oos_ind_recall']
        gross_result['eval_oos_ind_f_score'] = eval_result['oos_ind_f_score']
        gross_result['eval_eer'] = eval_result['eer']
        gross_result['eval_fpr95'] = ErrorRateAt95Recall(eval_result['all_binary_y'], eval_result['y_score'])
        gross_result['eval_auc'] = eval_result['auc']

        freeze_data['eval_result'] = eval_result

    if args.do_test:
        logger.info('#################### test result at step {} ####################'.format(global_step))
        if config['data_file'].startswith('binary'):
            # always keep test_set unchanged
            text_test_set = processor.read_dataset(data_path, ['test'])
        elif config['dataset'] == 'oos-eval':
            text_test_set = processor.read_dataset(data_path, ['test', 'oos_test'])
        elif config['dataset'] == 'smp':
            text_test_set = processor.read_dataset(data_path, ['test'])

        test_features = processor.convert_to_ids(text_test_set)

        if config['dataset'] == 'oos-eval':
            test_dataset = OOSDataset(test_features)
        if config['dataset'] == 'smp':
            test_dataset = SMPDataset(test_features)

        test_result = test(test_dataset)
        # logger.info(test_result)
        logger.info('test_eer: {}'.format(test_result['eer']))
        logger.info('test_ood_ind_precision: {}'.format(test_result['oos_ind_precision']))
        logger.info('test_ood_ind_recall: {}'.format(test_result['oos_ind_recall']))
        logger.info('test_ood_ind_f_score: {}'.format(test_result['oos_ind_f_score']))
        logger.info('test_auc: {}'.format(test_result['auc']))
        logger.info('test_fpr95: {}'.format(ErrorRateAt95Recall(test_result['all_binary_y'], test_result['y_score'])))
        my_plot_roc(test_result['all_binary_y'], test_result['y_score'],
                    os.path.join(args.output_dir, 'roc_curve.png'))
        save_result(test_result, os.path.join(args.output_dir, 'test_result'))
        # save_feature(test_result['all_features'], os.path.join(args.output_dir, 'test_feature'))
        gross_result['test_oos_ind_precision'] = test_result['oos_ind_precision']
        gross_result['test_oos_ind_recall'] = test_result['oos_ind_recall']
        gross_result['test_oos_ind_f_score'] = test_result['oos_ind_f_score']
        gross_result['test_eer'] = test_result['eer']
        gross_result['test_fpr95'] = ErrorRateAt95Recall(test_result['all_binary_y'], test_result['y_score'])
        gross_result['test_auc'] = test_result['auc']

        freeze_data['test_result'] = test_result

        # 输出错误cases
        if config['dataset'] == 'oos-eval':
            texts = [line[0] for line in text_test_set]
        elif config['dataset'] == 'smp':
            texts = [line['text'] for line in text_test_set]
        else:
            raise ValueError('The dataset {} is not supported.'.format(args.dataset))

        output_cases(texts, test_result['all_binary_y'], test_result['all_detection_binary_preds'],
                     os.path.join(args.output_dir, 'test_cases.csv'), processor)

        # confusion matrix
        plot_confusion_matrix(test_result['all_binary_y'], test_result['all_detection_binary_preds'],
                              args.output_dir)

        # beta_log_path = 'beta_log.txt'
        # if os.path.exists(beta_log_path):
        #     flag = True
        # else:
        #     flag = False
        # with open(beta_log_path, 'a', encoding='utf-8') as f:
        #     if flag == False:
        #         f.write('seed\tbeta\tdataset\tdev_eer\ttest_eer\tdata_size\n')
        #     line = '\t'.join([str(config['seed']), str(config['beta']), str(config['data_file']), str(best_dev), str(test_result['eer']), '100'])
        #     f.write(line + '\n')

        if args.do_vis:
            # [2 * length, feature_fim]
            features = np.concatenate([test_result['all_features'], get_fake_feature(len(test_dataset) // 2)], axis=0)
            features = TSNE(n_components=2, verbose=1, n_jobs=-1).fit_transform(features)  # [2 * length, 2]
            # [2 * length, １]
            if n_class > 2:
                labels = np.concatenate([test_result['all_y'], np.array([-1] * (len(test_dataset) // 2))], 0).reshape(
                    (-1, 1))
            else:
                labels = np.concatenate([test_result['all_binary_y'], np.array([-1] * (len(test_dataset) // 2))],
                                        0).reshape((-1, 1))
            # [2 * length, 3]
            data = np.concatenate([features, labels], 1)
            fig = scatter_plot(data, processor)
            fig.savefig(os.path.join(args.output_dir, 'plot.png'))
            fig.show()
            freeze_data['feature_label'] = data
            # plot_train_test(train_result['all_features'], test_result['all_features'], args.output_dir)

    with open(os.path.join(config['output_dir'], 'freeze_data.pkl'), 'wb') as f:
        pickle.dump(freeze_data, f)
    df = pd.DataFrame(data={'valid_y': freeze_data['valid_all_y'],
                            'valid_score': freeze_data['valid_score'],
                            })
    df.to_csv(os.path.join(config['output_dir'], 'valid_score.csv'))

    df = pd.DataFrame(data={'test_y': freeze_data['test_all_y'],
                            'test_score': freeze_data['test_score']
                            })
    df.to_csv(os.path.join(config['output_dir'], 'test_score.csv'))

    if args.result != 'no':
        pd_result = pd.DataFrame(gross_result)
        if args.seed == 16:
            pd_result.to_csv(args.result + '_gross_result.csv', index=False)
        else:
            pd_result.to_csv(args.result + '_gross_result.csv', index=False, mode='a', header=False)
        if args.seed == 8192:
            print(args.result)
            std_mean(args.result + '_gross_result.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # ------------------------data------------------------ #
    parser.add_argument('--dataset',
                        choices={'oos-eval', 'smp'}, required=True,
                        help='Which dataset will be used.')

    parser.add_argument('--data_file', required=False, type=str,
                        help="""Which type of dataset to be used, 
                        i.e. binary_undersample.json, binary_wiki_aug.json. Detail in config/data.ini""")
    # binary_smp_full base
    # binary_smp_full_v2 自己排除知识
    # binary_smp_full_v3 知识库排除
    # binary_smp_full_v4 知识库+ziji

    # ------------------------bert------------------------ #
    parser.add_argument('--bert_type',
                        choices={'bert-base-uncased', 'bert-large-uncased', 'bert-base-chinese', }, required=True,
                        help='Type of the pre-trained BERT to be used.')

    # ------------------------Discriminator------------------------ #
    parser.add_argument('--D_Wf_dim', default=512, type=int,
                        help='The Dimension of Wf for Discriminator.')

    parser.add_argument('--D_h_size', type=int, default=128, help='Number of hidden nodes in D.')

    # ------------------------Generator------------------------ #
    parser.add_argument('--G_z_dim', default=512, type=int,
                        help='The Dimension of z (noise) for Generator.')

    parser.add_argument('--G_h_size', type=int, default=128, help='Number of hidden nodes in G.')

    parser.add_argument('--feature_dim', default=768, type=int,
                        help='The Dimension of feature vector for Generator output and Discriminator input.')

    # ------------------------action------------------------ #
    parser.add_argument('--do_train', action='store_true',
                        help='Do training step')

    parser.add_argument('--do_eval', action='store_true',
                        help='Do evaluation on devset step')

    parser.add_argument('--do_test', action='store_true',
                        help='Do validation on testset step')

    parser.add_argument('--do_vis', action='store_true',
                        help='Do visualization.')

    parser.add_argument('--do_g_eval_vis', action='store_true',
                        help='Do visualization of generator and eval data feature.')

    parser.add_argument('--output_dir', required=True,
                        help='The output directory saving model and logging file.')

    parser.add_argument('--n_epoch', default=500, type=int,
                        help='Number of epoch for training.')

    parser.add_argument('--patience', default=10, type=int,
                        help='Number of epoch of early stopping patience.')

    parser.add_argument('--train_batch_size', default=64, type=int,
                        help='Batch size for training.')

    parser.add_argument('--predict_batch_size', default=32, type=int,
                        help='Batch size for evaluating and testing.')

    parser.add_argument('--D_lr', type=float, default=1e-5, help="Learning rate for Discriminator.")
    parser.add_argument('--G_lr', type=float, default=1e-5, help="Learning rate for Generator.")
    parser.add_argument('--beta', type=float, default=-1, help="Weight of fake sample loss for Discriminator.")

    parser.add_argument('--bert_lr', type=float, default=2e-4, help="Learning rate for Generator.")
    parser.add_argument('--fine_tune', action='store_true',
                        help='Whether to fine tune BERT during training.')
    parser.add_argument('--seed', type=int, default=123, help='seed')
    parser.add_argument('--model', type=str, required=True,
                        choices={'gan', 'dgan', 'lstm_gan', 'cnn_gan', 'realness_gan'},
                        help='choose gan model')
    parser.add_argument('--length_weight', type=float, default=0,
                        help="Weight of short and long sample loss for Discriminator.")
    parser.add_argument('--sample_weight', type=float, default=0,
                        help="Weight of excluded sample loss for Discriminator.")

    # data config
    parser.add_argument('--mode', type=int, default=-1, help="Controll the filtering way of knowledge sample")
    parser.add_argument('--maxlen', type=int, default=-1)
    parser.add_argument('--minlen', type=int, default=-1)
    parser.add_argument('--result', type=str, default="no")
    parser.add_argument('--ood', action='store_true', default=False)
    parser.add_argument('--optim_mode', type=int, default=0,
                        help="0: optimize both, and optimize without weight(pre-excluding samples); "
                             "1: optimize both, and only optimize length by weight; "
                             "2: optimize both, and only optimize sample by weight;"
                             "3: optimize both, and optimize both by weight")

    # RealnessGAN
    parser.add_argument('--D_updates', type=int, default=1, help='Number of D updating per iteration cycle.')
    parser.add_argument('--G_updates', type=int, default=1, help='Number of G updating per iteration cycle.')

    parser.add_argument('--adam_eps', type=float, default=1e-08, help='Adam eps.')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam betas[0].')
    parser.add_argument('--beta2', type=float, default=0.999, help='Adam betas[1].')
    parser.add_argument('--decay', type=float, default=0, help='Decay to apply to lr each cycle. decay^n_iter gives the final lr.')
    parser.add_argument('--weight_decay', type=float, default=0, help='L2 regularization weight. Helps convergence but leads to artifacts in images, not recommended.')

    parser.add_argument('--positive_skew', type=float, default=1.0, help='Skewness of anchor1 when computing loss.')
    parser.add_argument('--negative_skew', type=float, default=-1.0, help='Skewness of anchor0 when computing loss.')
    parser.add_argument('--num_outcomes', type=int, default=20, help='Number of outcomes of D.')
    parser.add_argument('--relativisticG', action='store_true', default=False, help='Whether to use relativistic trick when training G.')
    parser.add_argument('--use_adaptive_reparam', action='store_true', default=False, help='Whether to use re-parameterization trick in training.')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logger = Logger(os.path.join(args.output_dir, 'train.log'))
    main(args)
