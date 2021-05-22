# coding: utf-8
# @author: Ross

from processor.base_processor import BertProcessor

import os
import json
from config import Config
from configparser import SectionProxy
import numpy as np
from scipy import stats

class SMPProcessor_v3(BertProcessor):
    """oos-eval 数据集处理"""

    def __init__(self, bert_config, maxlen=32):
        super(SMPProcessor_v3, self).__init__(bert_config, maxlen)

    def convert_to_ids(self, dataset: list) -> list:
        ids_data = []
        print('dataset', type(dataset))
        for line in dataset:
            ids_data.append(self.parse_line(line))
        return ids_data

    def read_dataset(self, path: str, data_types: list, mode=0, maxlen=-1, minlen=-1, pre_exclude=False):
        """
        读取数据集文件
        :param path: 路径
        :param data_types:  [type1, type2]
        :param mode: 读取模式
        :param maxlen: 最大长度
        :return:
        """
        self.mode = mode
        dataset = []
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for data_type in data_types:
            for line in data[data_type]:
                if pre_exclude:
                    if maxlen != -1 and len(line['text']) > maxlen:
                        continue
                    if minlen != -1 and len(line['text']) <= minlen:
                        continue
                    if mode == 1 and line['knowledge'] == 1:
                        continue
                    if mode == 2 and line['knowledge'] == 2:
                        continue
                    if mode == 3 and line['knowledge'] != 0:
                        continue
                dataset.append(line)
        return dataset

    def load_label(self, path):
        """load label"""
        with open(path, 'r', encoding='utf-8') as f:
            self.id_to_label = json.load(f)
            self.label_to_id = {label: i for i, label in enumerate(self.id_to_label)}

    def parse_line(self, line: dict) -> list:
        """
        :param line: [text, label]
        :return: [text_ids, mask, type_ids, knowledge_tag, label_ids]
        """
        text = line['text']
        label = line['domain']
        # knowledge tag
        if 'knowledge' in line and self.mode != 0:
            knowledge_tag = line['knowledge']
        else:
            knowledge_tag = 0

        ids = self.parse_text_to_bert_token(text) + [knowledge_tag] + [self.parse_label(label)]
        return ids

    def parse_text(self, text) -> (list, list, list):
        """
        将文本转为ids
        :param text: 字符串文本
        :return: [token_ids, mask, type_ids]
        """
        return self.parse_text_to_bert_token(text)

    def parse_label(self, label):
        """
        讲label转为ids
        :param label: 文本label
        :return: ids
        """
        return self.label_to_id[label]

    def remove_minlen(self, dataset, minlen):
        n_dataset = []
        for i, line in enumerate(dataset):
            if len(line['text']) >= minlen:
                n_dataset.append(line)
        return n_dataset

    def remove_maxlen(self, dataset, maxlen):
        n_dataset = []
        for i, line in enumerate(dataset):
            if len(line['text']) <= maxlen:
                n_dataset.append(line)
        return n_dataset

    def get_smp_data_info(self, data_path):
        """

        Args:
            data_path: url

        Returns: {'train':{'num':..., 'ood':..., 'id':..., 'tex_len': [(), ()], 'all_len': [],
                  'val':....,
                  'test':...}

        """
        result = {}
        with open(data_path, 'r', encoding='utf-8') as fp:
            source = json.load(fp)
            for type in source:
                n = 0
                n_id = 0
                n_ood = 0
                text_len = {}
                all_text_len = []
                for line in source[type]:
                    if line['domain'] == 'chat':
                        n_ood += 1
                    else:
                        n_id += 1
                    n += 1
                    text_len[len(line['text'])] = text_len.get(len(line['text']), 0) + 1
                    all_text_len.append(len(line['text']))
                result[type] = {'num': n, 'ood': n_ood, 'id': n_id,
                                'text_len': sorted(text_len.items(), key=lambda d: d[0], reverse=False),
                                'all_len': all_text_len}
        return result

    def get_conf_intveral(self, data: list, alpha, logarithm=False):
        """
        置信区间
        Args:
            data:

        Returns: (a, b)
        a: 置信上界; b: 置信下界

        alpha : array_like of float
            Probability that an rv will be drawn from the returned range.
            Each value should be in the range [0, 1].
        """
        data = np.array(data)
        if logarithm:   # 对数正态分布
            data = np.log(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        # from scipy import stats
        skew = stats.skew(data)  # 求偏度
        kurtosis = stats.kurtosis(data)  # 求峰度
        conf_intveral = stats.norm.interval(alpha, loc=mean, scale=std) # 置信区间
        if logarithm:
            conf_intveral = np.exp(conf_intveral)
        return conf_intveral



# --------------------oos-eval-------------------- #

def preprocess_smp_eval(config: SectionProxy):
    data_dir = config['DataDir']
    files = os.listdir(data_dir)
    for file in files:
        # if not data file, skip
        if not file.endswith('.json'):
            continue

        # if output file exists, skip
        output_file = file.replace('.json', '.label')
        if os.path.exists(os.path.join(data_dir, output_file)):
            continue

        labels = set()
        with open(os.path.join(data_dir, file), 'r', encoding='utf-8') as f:
            data = json.load(f)
        # for k, v in data.items():
        #     for line in v:
        #         labels.add(line[1])
        for line in data['train']:
            labels.add(line['domain'])

        # 对类按照字典序排序后，将oos放在最后面
        try:
            labels.remove('chat')
        except KeyError:
            pass

        labels = ['chat'] + sorted(labels)

        with open(os.path.join(data_dir, output_file), 'w', encoding='utf-8') as f:
            json.dump(labels, f, ensure_ascii=False, indent=2)


config = Config('config/data.ini')
oos_config = config('smp')
preprocess_smp_eval(oos_config)

# --------------------oos-eval-------------------- #
