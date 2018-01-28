import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import sys
sys.path.append('/home/xiaox/studio/src/git-series')
import sigma
from sigma import layers, helpers, ops
import tensorflow as tf
import numpy as np
import random
import json

import argparse

def load(jsonfile):
    with open(jsonfile) as jf:
        intents = json.load(jf)
    words = []
    classes = []
    documents = []
    ignore_words = ['?']
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            w = nltk.word_tokenize(pattern)
            words.extend(w)
            documents.append((w, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])
    words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
    words = sorted(list(set(words)))
    classes = sorted(list(set(classes)))
    print('{} documents'.format(len(documents)))
    print('{} classes'.format(len(classes)))
    #print('{} unique stemmed words'.format(words))
    training = []
    outout = []
    output_empty = [0] * len(classes)
    for doc in documents:
        bag = []
        pattern_words = doc[0]
        pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
        for w in words:
            bag.append(1) if w in pattern_words else bag.append(0)
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1
        training.append([bag, output_row])
    np.random.shuffle(training)
    training = np.array(training)
    return list(training[:, 0]), list(training[:, 1]), words, classes, intents


def _clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [stemmer.stem(word.lower()) for word in sentence_words]


def bag_of_words(sentence, words):
    sentence_words = _clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)


def build(input_shape, nclass):
    inputs = sigma.placeholder(dtype='float32', shape=input_shape, name='input-data')
    with sigma.defaults(act='relu'):
        x = layers.convs.dense(inputs, 8)
        x = layers.convs.dense(x, 8)
        x = layers.convs.dense(x, nclass, act=None)
    return inputs, x


def predict(dataset, sentence, checkpoints):
    x, y, words, classes, intents = load(dataset)
    xtensor, preds = build([None, len(x[0])], nclass=len(y[0]))
    ans = sigma.session(checkpoints=checkpoints)
    sess = ans['session']
    presentation = bag_of_words(sentence, words)
    presentation = presentation.reshape([1, -1])
    _preds = sigma.predict(ops.helper.shape(preds))(preds)
    _x = sess.run(_preds, feed_dict={xtensor:presentation})
    sess.close()
    return _x, classes, intents


def classify(dataset, sentence, checkpoints):
    thresh = 0.25
    result, classes, intents = predict(dataset, sentence, checkpoints)
    result = [[i, r] for i, r in enumerate(result) if r > thresh]
    result.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in result:
        return_list.append((classes[r[0]], r[1]))
    return return_list, intents


def response(dataset, sentence, checkpoints, user_id='121'):
    result, intents = classify(dataset, sentence, checkpoints)
    context = {}
    print(sentence, '==>')
    if result:
        while result:
            for i in intents['intents']:
                if i['tag'] == result[0][0]:
                    if 'context_set' in i:
                        context[user_id] = i['context_set']
                    if not 'context_filter' in i or \
                        (user_id in context and 'context_filter' in i and i['context_filter'] == context[user_id]):
                        return print('[SIGMA]:', random.choice(i['responses']))
            result.pop(0)


def train(dataset, checkpoints, epochs=50, batch_size=8, shuffle=True):
    x, y, *_ = load(dataset)
    xtensor, preds = build([None, len(x[0])], nclass=len(y[0]))
    #helpers.export_graph('chatbot.png')
    ytensor = sigma.placeholder(dtype='int32', shape=[None, len(y[0])])
    loss = layers.losses.cce(preds, ytensor, onehot=True)
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

    sigma.run(x, xtensor, optimizer, loss, y, ytensor,
              epochs=epochs,
              batch_size=batch_size,
              shuffle=shuffle,
              checkpoints=checkpoints,
              logs='cache/logs',
              save='min')

parser = argparse.ArgumentParser()
parser.add_argument('--train', type=bool, default=False)
parser.add_argument('--sentence', type=str, default='is your shop open today?')
parser.add_argument('--checkpoints', type=str, default='cache/checkpoints/model')
parser.add_argument('--dataset', type=str, default='db/intents.json')

if __name__ == '__main__':
    args = parser.parse_args()
    if args.train:
        train(args.dataset, args.checkpoints)
    else:
        response(args.dataset, args.sentence, args.checkpoints)
