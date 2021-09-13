import os
import shutil
import logging
import argparse
import random
import re
import codecs
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict
from GBDT.gbdt import GradientBoostingBinaryClassifier


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.removeHandler(logger.handlers[0])
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)


def pos_process(data, seq2lrc):
    res = []
    for line in tqdm(data):
        index, hot, ids, first, second, label, lrc = line
        count = 1e-6
        first_score = second_score = 0
        max_first = max_second = float('-inf')
        min_first = min_second = float('inf')
        lrc = lrc[2:]
        first_seq = lrc[0]
        second_seq = lrc[1]
        for seq in lrc:
            if seq not in seq2lrc.keys():
                continue

            if second_seq in seq2lrc.keys() and first_seq in seq2lrc.keys():
                is_same_second_seq = (len(seq2lrc[second_seq] & seq2lrc[seq]) - 1) / (len(seq2lrc[second_seq]) +
                                                                                      len(seq2lrc[seq]) - 2 + 1e-6)
                is_same_first_seq = (len(seq2lrc[first_seq] & seq2lrc[seq]) - 1) / (len(seq2lrc[first_seq]) +
                                                                                    len(seq2lrc[seq]) - 2 + 1e-6)
                count += 1
                max_first = max(max_first, is_same_first_seq)
                max_second = max(max_second, is_same_second_seq)
                min_first = min(min_first, is_same_first_seq)
                min_second = min(min_second, is_same_second_seq)
                first_score += is_same_first_seq
                second_score += is_same_second_seq
            first_seq = second_seq
            second_seq = seq
        res.append([index, hot, ids, first_score / count, second_score / count, count, max_first, min_first, max_second, min_second, 1])
    return res


def data_process(file_path):

    special = ['10亿现金激励千亿流量扶持', '本歌曲来自〖网易音乐人〗', '版权所有未经许可请勿使用', '本歌曲来自〖网易飓风计划〗',
              '未经著作权人许可不得翻唱翻录或使用', '本歌曲来自〖云上工作室〗', '本歌曲来自〖飓风计划〗', '北京好听音乐录音棚',
              ]

    positive = []
    lines = []
    # 去除名字中含有“串烧"的歌曲
    with codecs.open(file_path) as fr:
        for line in tqdm(fr):
            line = line.strip().split('\001')
            index, name, hot, lrc, ids = line
            hot = int(hot)
            ids = len(ids.strip().split('\002'))
            lrc = re.split('[，。]', lrc)
            if '串烧' in name or '+' in name or '×' in name or '串烧' in lrc[0]:
                positive.append([index, hot, ids, 0, 0, 1, lrc])
                continue
            lines.append([index, hot, ids, 0, 0, 1, lrc])

    # 建立字典
    seq2lr = {}
    index = 0
    for line in tqdm(lines):
        indx, hot, ids, first, second, label, lrc = line
        for seq in lrc:
            if len(seq) <= 6 or len(set(seq.replace(' ', ''))) <= 6 or seq in special:
                continue
            if seq not in seq2lr.keys():
                seq2lr[seq] = set()
            seq2lr[seq].add(int(index))
        index += 1

    seq2lrc = OrderedDict(sorted(seq2lr.items(), key=lambda x: len(x[1]), reverse=True))

    # 获取训练语料
    positive = pos_process(positive, seq2lrc)
    negative = pos_process(lines, seq2lrc)
    test = pos_process(lines, seq2lrc)

    random.shuffle(positive)
    random.shuffle(negative)
    random.shuffle(test)

    train_data = positive[:1500] + negative[:1500]
    # test_pos_neg = positive[1500:] + negative[1500:len(positive)]
    with open('/mnt/dl-storage/dg-cephfs-0/group/ai-nlp/houtongpeng/LrcSearch/move_skewer_plus.txt', 'r', encoding='utf8') as fr:
        skewer = []
        for line in tqdm(fr):
            line = line.strip().split('\001')
            index, name, hot, lrc, ids = line
            hot = int(hot)
            ids = len(ids.strip().split('\002'))
            lrc = re.split('[，。]', lrc)
            skewer.append([index, hot, ids, 0, 0, 1, lrc])
    test_pos_neg = pos_process(skewer, seq2lrc)
    test_part = test[:10000]

    random.shuffle(train_data)
    random.shuffle(test_pos_neg)

    return train_data, test_pos_neg, test_part, test


def run(args):
    # 获取训练和测试数据
    data = pd.read_csv('data/train_data.csv', encoding='utf8', sep=',')
    test_data = pd.read_csv('data/test_data.csv', encoding='utf8', sep=',')

    # data, test_pos_neg, test_part, test = data_process(args.path)
    # data = pd.DataFrame(data=data, columns=['index', 'hot', 'ids', 'first', 'second', 'count', 'max_first', 'min_first', 'max_second', 'min_second', 'label'])
    # test_data = pd.DataFrame(data=test, columns=['index', 'hot', 'ids', 'first', 'second', 'count', 'max_first', 'min_first', 'max_second', 'min_second', 'label'])

    data = data.iloc[:, 1:]
    # data = data.loc[:, ['hot', 'ids', 'first', 'second', 'count', 'max_first', 'min_first', 'max_second', 'min_second', 'label']]

    labels = pd.concat([test_data['index'], test_data['label']], axis=1)

    test_data = test_data.iloc[:, 1:-1]
    # test_data = test_data.loc[:, ['hot', 'ids', 'first', 'second', 'count', 'max_first', 'min_first', 'max_second', 'min_second']]

    # 创建模型结果的目录
    if not os.path.exists('results'):
        os.makedirs('results')
    if len(os.listdir('results')) > 0:
        shutil.rmtree('results')
        os.makedirs('results')
    # 初始化模型
    model = GradientBoostingBinaryClassifier(learning_rate=args.lr, n_trees=args.trees, max_depth=args.depth,
                                             is_log=args.log)
    # 训练模型
    model.fit(data)
    # 记录日志
    logger.removeHandler(logger.handlers[-1])
    logger.addHandler(logging.FileHandler('results/result.log'.format(iter), mode='w', encoding='utf-8'))
    logger.info(data)
    # 模型预测
    model.predict(test_data)
    # 记录日志
    logger.setLevel(logging.INFO)
    logger.info((test_data['predict_proba']))
    logger.info((test_data['predict_label']))

    count = 0
    TP = 0
    FN = 0
    p = []
    r = []
    neg = []
    pos = []
    for prob, label, index in zip(test_data['predict_label'], labels['label'], labels['index']):
        if prob == 1:
            pos.append(index)
            count += 1
        if prob == 0:
            neg.append(index)
        if prob == 1 and label == 1:
            TP += 1
        if prob == 1 and label == 0:
            p.append(index)
        if prob == 0 and label == 1:
            r.append(index)
            FN += 1

    print('精确率：', TP / count)
    print('召回率：', TP / (TP + FN))
    print(sorted(neg))
    print(len(neg))
    print(sorted(pos))
    print(len(pos))
    print('-' * 100)
    print(sorted(p))
    print(len(p))
    # print(sorted(r))
    # print(len(r))
    # 精确率：0.9520
    # 召回率：0.8792


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GBDT-Simple-Tutorial')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--trees', default=5, type=int, help='the number of decision trees')
    parser.add_argument('--depth', default=3, type=int, help='the max depth of decision trees')
    # 非叶节点的最小数据数目，如果一个节点只有一个数据，那么该节点就是一个叶子节点，停止往下划分
    parser.add_argument('--count', default=2, type=int, help='the min data count of a node')
    parser.add_argument('--log', default=True, type=bool, help='whether to print the log on the console')
    parser.add_argument('--path', default='', type=bool, help='data path')
    parser.add_argument('--topath', default='', type=bool, help='save data path')
    args = parser.parse_args()
    run(args)
