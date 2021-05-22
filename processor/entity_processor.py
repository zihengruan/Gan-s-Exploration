# -*- coding: utf-8 -*-

"""
@author: YuKI
@contact: 1162236967@qq.com
@file: entity_processor
@time: 2021/1/26 9:30
"""

import os
import pandas as pd
import ast
import re
import json


class EntityProcessor:
    def __init__(self, filepath, mode):
        if mode == 1:
            self.data = self.load_entity(filepath)
        else:
            self.data = self.load_entity_2(filepath)
        self.compiled = []
        self.compile()

    def load_entity(self, filepath):
        df = pd.read_excel(filepath)
        values = df.values
        t_data = []
        for line in values:
            each = list(ast.literal_eval(line[2]).values())
            if len(each) > 0:
                each = list(each[0])[0]
                if each not in t_data:
                    t_data.append(each)
        return t_data

    def load_entity_2(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as fp:
            entity = json.load(fp)
        return entity

    def remove_entity(self, text):
        for line in self.compiled:
            text = re.sub(line, '', text)
        return text

    def compile(self):
        for line in self.data:
            self.compiled.append(re.compile(line))

    def remove_smp_entity(self, dataset):
        num = 0
        for line in dataset:
            previous_len = line['text']
            line['text'] = self.remove_entity(line['text'])
            solved_len = line['text']
            if previous_len > solved_len:
                num += 1
        return dataset, num


if __name__ == '__main__':
    filepath = '../data/smp/训练集 全知识标记.xlsx'
    entityProcessor = EntityProcessor(filepath)
    data = entityProcessor.data
    print(data)
    print(len(data))
    entityProcessor.compile()
    print(entityProcessor.compiled)
    r1 = re.compile('qq')
    r2 = re.compile('123')
    text1 = '打开qq'
    m1 = re.sub(r1, '', text1)
    print(m1)
    m2 = re.sub(r2, '', text1)
    print(m2)
    print(entityProcessor.remove_entity('打开qq 不加 平方 上 应用 十八'))
    dataset = [{'text': '123'}]
    for line in dataset:
        line['text'] = '456'
    print(dataset)

