# coding: utf-8
# @author: Ross
# @file: tool.py 
# @time: 2020/01/14
# @contact: devross@gmail.com
import json
import os
import random
from sklearn.metrics import roc_auc_score
import numpy as np

import numpy
import pandas as pd
import torch

if torch.cuda.is_available():
    device = 'cuda:0'
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
else:
    device = 'cpu'
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor


def check_manual_seed(seed):
    """ If manual seed is not specified, choose a random one and communicate it to the user."""

    seed = seed or random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)
    # 添加随机数种子
    numpy.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False

    print('Using manual seed: {seed}'.format(seed=seed))


def convert_to_int_by_threshold(array, threshold=0.5):
    """
    根据阈值，将浮点数转为0/1
    :param array: numpy数组或者pytorch.Tensor对象
    :param threshold: 阈值
    :return: 0/1数组
    """
    _array = array.numpy() if isinstance(array, torch.Tensor) else array
    _array = (_array > threshold).astype(int)
    return _array


def output_cases(texts, ground_truth, predicts, path, processor, logit=None):
    """
    生成csv case
    :param texts: 文本 list
    :param ground_truth: 真正标签 数组
    :param predicts: 预测 数组
    :param path: 保存路径
    :param processor: 数据集处理器
    :return:
    """
    ground_truth = ground_truth.numpy().astype(int) if isinstance(ground_truth, torch.Tensor) else ground_truth
    predicts = predicts.numpy().astype(int) if isinstance(
        predicts, torch.Tensor) else predicts
    if logit != None:
        df = pd.DataFrame(data={
            'text': texts,
            'ground_truth': ground_truth,
            'predict': predicts,
            'score': logit
        })
    else:
        df = pd.DataFrame(data={
            'text': texts,
            'ground_truth': ground_truth,
            'predict': predicts
        })
    df['ground_truth_label'] = df['ground_truth'].apply(lambda i: processor.id_to_label[i])
    df['predict_label'] = df['predict'].apply(lambda i: processor.id_to_label[i])
    df['ground_truth_is_ind'] = df['ground_truth'] != 0
    df['predict_is_ind'] = df['predict'] != 0

    os.makedirs(os.path.dirname(path), exist_ok=True)
    if logit == None:
        df.to_csv(path, columns=['text',
                                 'ground_truth', 'predict',
                                 'ground_truth_label', 'predict_label',
                                 'ground_truth_is_ind', 'predict_is_ind'],
                  index=False)
    else:
        df.to_csv(path, columns=['text',
                                 'score',
                                 'ground_truth', 'predict',
                                 'ground_truth_label', 'predict_label',
                                 'ground_truth_is_ind', 'predict_is_ind'],
                  index=False)

def save_gan_model(discriminator: torch.nn.Module, generator: torch.nn.Module, path):
    """保存GAN模型"""
    state_dict = {'discriminator': discriminator.state_dict(),
                  'generator': generator.state_dict()
                  }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state_dict, path)
    return state_dict


def load_gan_model(discriminator: torch.nn.Module, generator: torch.nn.Module, path: str):
    """加载GAN模型"""
    checkpoint = torch.load(path)
    discriminator.load_state_dict(checkpoint['discriminator'])
    generator.load_state_dict(checkpoint['generator'])
    return discriminator, generator

def save_realness_gan_model(realness_discriminator: torch.nn.Module, classification_discriminator: torch.nn.Module, generator: torch.nn.Module, path):
    """保存realness GAN模型"""
    state_dict = {'realness_discriminator': realness_discriminator.state_dict(),
                  'classification_discriminator': classification_discriminator.state_dict(),
                  'generator': generator.state_dict()
                  }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state_dict, path)
    return state_dict


def load_realness_gan_model(realness_discriminator: torch.nn.Module, classification_discriminator: torch.nn.Module, generator: torch.nn.Module, path: str):
    """加载realness GAN模型"""
    checkpoint = torch.load(path)
    realness_discriminator.load_state_dict(checkpoint['realness_discriminator'])
    classification_discriminator.load_state_dict(checkpoint['classification_discriminator'])
    generator.load_state_dict(checkpoint['generator'])
    return realness_discriminator, classification_discriminator, generator


def save_model(model: torch.nn.Module, path: str, model_name: str):
    '''保存模型'''
    state_dict = {model_name: model.state_dict()}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state_dict, path)
    return state_dict


def load_model(model: torch.nn.Module, path: str, model_name: str):
    '''保存模型'''
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint[model_name])
    return model


class MyEncoder(json.JSONEncoder):
    '''
    保存为json文件时的格式转换
    '''

    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.
    refer to: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    """

    def __init__(self, patience=7, delta=0, logger=None):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """

        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.logger = logger

    def __call__(self, score):
        """
        return:
         1 表示要保存模型
         0 表示不需要保存模型
         -1 表示不需要模型，且超过了patience，需要early stop
        """
        if self.best_score is None:
            if self.logger:
                self.logger.info('Saving model, best score is {}'.format(score))
            self.best_score = score
            return 1
        elif score <= self.best_score + self.delta:
            self.counter += 1
            if self.logger:
                self.logger.info('EarlyStopping counter: {} out of {}, best score is {}'.
                                 format(self.counter, self.patience, self.best_score))
            if self.counter >= self.patience:
                if self.logger:
                    self.logger.info('Stopping training.')
                return -1
            return 0
        else:
            if self.logger:
                self.logger.info('Saving model, best score is {}, improved by {}'.format(score, score - self.best_score))
            self.best_score = score
            self.counter = 0
            return 1


def mask_ood(p_tensor: torch.Tensor) -> torch.Tensor:
    batch, n_class = p_tensor.shape
    mask = torch.zeros((batch, n_class))
    mask[:, 0] = -1e32
    mask = torch.FloatTensor(mask)
    print(p_tensor.shape)
    print(mask.shape)
    return p_tensor + mask

def roc(y_true, y_scores):
    return roc_auc_score(y_true, y_scores)

def ErrorRateAt95Recall(labels, scores):
    recall_point = 0.95
    labels = numpy.asarray(labels)
    scores = numpy.asarray(scores)
    # Sort label-score tuples by the score in descending order.
    indices = numpy.argsort(scores)[::-1]    #降序排列
    sorted_labels = labels[indices]
    sorted_scores = scores[indices]
    n_match = sum(sorted_labels)
    n_thresh = recall_point * n_match
    thresh_index = numpy.argmax(numpy.cumsum(sorted_labels) >= n_thresh)
    FP = numpy.sum(sorted_labels[:thresh_index] == 0)
    TN = numpy.sum(sorted_labels[thresh_index:] == 0)
    return float(FP) / float(FP + TN)

def save_result(result, save_path):
    numpy.save(save_path, numpy.array([result['all_binary_y'], result['y_score']]))

def save_feature(feature, save_path):
    numpy.save(save_path, numpy.array(feature))

def std_mean(path):
    std_result = {}
    mean_result = {}
    std_result['std'] = ['std_ood', 'std_in']
    mean_result['mean'] = ['mean_ood', 'mean_id']
    all_gross_result = pd.read_csv(path)
    all_eval_oos_precision = all_gross_result['eval_oos_ind_precision'].values[0::2]
    all_eval_ind_precision = all_gross_result['eval_oos_ind_precision'].values[1::2]
    all_eval_oos_recall = all_gross_result['eval_oos_ind_recall'].values[0::2]
    all_eval_ind_recall = all_gross_result['eval_oos_ind_recall'].values[1::2]
    all_eval_oos_f_score = all_gross_result['eval_oos_ind_f_score'].values[0::2]
    all_eval_ind_f_score = all_gross_result['eval_oos_ind_f_score'].values[1::2]
    all_eval_eer = all_gross_result['eval_eer'].values[0::2]
    all_eval_fpr95 = all_gross_result['eval_fpr95'].values[0::2]
    all_eval_auc = all_gross_result['eval_auc'].values[0::2]
    mean_eval_oos_ind_precision = [np.mean(np.array(all_eval_oos_precision)), np.mean(np.array(all_eval_ind_precision))]
    mean_eval_oos_ind_recall = [np.mean(np.array(all_eval_oos_recall)), np.mean(np.array(all_eval_ind_recall))]
    mean_eval_oos_ind_f_score = [np.mean(np.array(all_eval_oos_f_score)), np.mean(np.array(all_eval_ind_f_score))]
    mean_eval_eer = np.mean(np.array(all_eval_eer))
    mean_eval_fpr95 = np.mean(np.array(all_eval_fpr95))
    mean_eval_auc = np.mean(np.array(all_eval_auc))
    std_eval_oos_ind_precision = [np.std(np.array(all_eval_oos_precision)), np.std(np.array(all_eval_ind_precision))]
    std_eval_oos_ind_recall = [np.std(np.array(all_eval_oos_recall)), np.std(np.array(all_eval_ind_recall))]
    std_eval_oos_ind_f_score = [np.std(np.array(all_eval_oos_f_score)), np.std(np.array(all_eval_ind_f_score))]
    std_eval_eer = np.std(np.array(all_eval_eer))
    std_eval_fpr95 = np.std(np.array(all_eval_fpr95))
    std_eval_auc = np.std(np.array(all_eval_auc))
    std_result['std_eval_oos_ind_precision'] = std_eval_oos_ind_precision
    std_result['std_eval_oos_ind_recall'] = std_eval_oos_ind_recall
    std_result['std_eval_oos_ind_f_score'] = std_eval_oos_ind_f_score
    std_result['std_eval_eer'] = std_eval_eer
    std_result['std_eval_fpr95'] = std_eval_fpr95
    std_result['std_eval_auc'] = std_eval_auc
    mean_result['mean_eval_oos_ind_precision'] = mean_eval_oos_ind_precision
    mean_result['mean_eval_oos_ind_recall'] = mean_eval_oos_ind_recall
    mean_result['mean_eval_oos_ind_f_score'] = mean_eval_oos_ind_f_score
    mean_result['mean_eval_eer'] = mean_eval_eer
    mean_result['mean_eval_fpr95'] = mean_eval_fpr95
    mean_result['mean_eval_auc'] = mean_eval_auc

    all_test_oos_precision = all_gross_result['test_oos_ind_precision'].values[0::2]
    all_test_ind_precision = all_gross_result['test_oos_ind_precision'].values[1::2]
    all_test_oos_recall = all_gross_result['test_oos_ind_recall'].values[0::2]
    all_test_ind_recall = all_gross_result['test_oos_ind_recall'].values[1::2]
    all_test_oos_f_score = all_gross_result['test_oos_ind_f_score'].values[0::2]
    all_test_ind_f_score = all_gross_result['test_oos_ind_f_score'].values[1::2]
    all_test_eer = all_gross_result['test_eer'].values[0::2]
    all_test_fpr95 = all_gross_result['test_fpr95'].values[0::2]
    all_test_auc = all_gross_result['test_auc'].values[0::2]
    mean_test_oos_ind_precision = [np.mean(np.array(all_test_oos_precision)), np.mean(np.array(all_test_ind_precision))]
    mean_test_oos_ind_recall = [np.mean(np.array(all_test_oos_recall)), np.mean(np.array(all_test_ind_recall))]
    mean_test_oos_ind_f_score = [np.mean(np.array(all_test_oos_f_score)), np.mean(np.array(all_test_ind_f_score))]
    mean_test_eer = np.mean(np.array(all_test_eer))
    mean_test_fpr95 = np.mean(np.array(all_test_fpr95))
    mean_test_auc = np.mean(np.array(all_test_auc))
    std_test_oos_ind_precision = [np.std(np.array(all_test_oos_precision)), np.std(np.array(all_test_ind_precision))]
    std_test_oos_ind_recall = [np.std(np.array(all_test_oos_recall)), np.std(np.array(all_test_ind_recall))]
    std_test_oos_ind_f_score = [np.std(np.array(all_test_oos_f_score)), np.std(np.array(all_test_ind_f_score))]
    std_test_eer = np.std(np.array(all_test_eer))
    std_test_fpr95 = np.std(np.array(all_test_fpr95))
    std_test_auc = np.std(np.array(all_test_auc))
    std_result['std_test_oos_ind_precision'] = std_test_oos_ind_precision
    std_result['std_test_oos_ind_recall'] = std_test_oos_ind_recall
    std_result['std_test_oos_ind_f_score'] = std_test_oos_ind_f_score
    std_result['std_test_eer'] = std_test_eer
    std_result['std_test_fpr95'] = std_test_fpr95
    std_result['std_test_auc'] = std_test_auc
    mean_result['mean_test_oos_ind_precision'] = mean_test_oos_ind_precision
    mean_result['mean_test_oos_ind_recall'] = mean_test_oos_ind_recall
    mean_result['mean_test_oos_ind_f_score'] = mean_test_oos_ind_f_score
    mean_result['mean_test_eer'] = mean_test_eer
    mean_result['mean_test_fpr95'] = mean_test_fpr95
    mean_result['mean_test_auc'] = mean_test_auc

    print(std_result)
    print(mean_result)
    std_result = pd.DataFrame(std_result)
    mean_result = pd.DataFrame(mean_result)
    std_result.to_csv(path, index=False, mode='a', header=False)
    mean_result.to_csv(path, index=False, mode='a', header=False)
