# -*- coding: utf-8 -*-
# @Time    : 2020/12/18 5:57 下午
# @Author  : Jiangweiwei
# @mail    : zhongerqiandan@163.com

import numpy as np
from tqdm import tqdm
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, open
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
from keras.models import Model
import os
from random import sample, shuffle
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 基本参数
maxlen = 128
batch_size = 64
epochs = 11

# bert配置
config_path = '/data/jiangweiwei/bertmodel/chinese_wwm_ext/bert_config.json'
checkpoint_path = '/data/jiangweiwei/bertmodel/chinese_wwm_ext/bert_model.ckpt'
dict_path = '/data/jiangweiwei/bertmodel/chinese_wwm_ext/vocab.txt'
model_save_path = '/data/jiangweiwei/models/couplet/best_model.weights'

# 加载并精简词表，建立分词器
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)



class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分
    """

    def compute_loss(self, inputs, mask=None):
        y_true, y_mask, y_pred = inputs
        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = y_mask[:, 1:]  # segment_ids，刚好指示了要预测的部分
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss


model = build_transformer_model(
    config_path,
    checkpoint_path,
    application='unilm',
    keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
)

output = CrossEntropy(2)(model.inputs + model.outputs)

model = Model(model.inputs, output)
model.compile(optimizer=Adam(1e-5))
model.summary()


class Couplet(AutoRegressiveDecoder):
    """seq2seq解码器
    """

    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        return model.predict([token_ids, segment_ids])[:, -1]

    def generate(self, text, topk=1):
        max_c_len = maxlen - self.maxlen
        token_ids, segment_ids = tokenizer.encode(text, maxlen=max_c_len)
        output_ids = self.beam_search([token_ids, segment_ids],
                                      topk)  # 基于beam search
        return tokenizer.decode(output_ids)

    def generate_random_sample(self, text, return_num=1, topk=5):
        max_c_len = maxlen - self.maxlen
        token_ids, segment_ids = tokenizer.encode(text, maxlen=max_c_len)
        output_ids = self.random_sample([token_ids, segment_ids], return_num, topk=topk)
        sentences = []
        for ids in output_ids:
            sentences.append(tokenizer.decode(ids))
        return sentences


couplet = Couplet(start_id=None, end_id=tokenizer._token_end_id, maxlen=64)


if __name__ == '__main__':

    model.load_weights(model_save_path)
    text = input('输入：')
    while 1:
        gen = couplet.generate_random_sample(text.strip(), topk=10)
        print(f'输出：{gen[0]}')
        print('-' * 36)
        text = input('输入：')
