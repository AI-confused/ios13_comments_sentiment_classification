import codecs
import pandas as pd
import numpy as np
import argparse
import jieba
import os


def train_label_process():
    train = pd.read_csv('data/train.csv', index_col=False, encoding='utf-8')
    return train

def is_chinese(uchar):
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False

def is_punctuation(uchar):
    punctuations = ['，', '。', '？', '！', '：']
    if uchar in punctuations:
        return True
    else:
        return False

    
def is_stopword(word, stopwords):
    if word in stopwords:
        return True
    else:
        return False
    
    
def text_process(text, punc, flag, stopwords):
    temp_text = ''
    for _ in text:
        if punc:
            if is_chinese(_):
                temp_text += _
        else:
            if is_chinese(_) or is_punctuation(_):
                temp_text += _
    text = temp_text
    
    if flag: 
        words = list(jieba.cut(temp_text))
#         print(words)
        text_temp = ''
        for _ in words:
            if not is_stopword(_, stopwords):
                text_temp += _
        text = text_temp
#     print(text)
    return text

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='clean data')
    parser.add_argument('-stop-word', type=int, default=1, help='delete stop words in text')
    parser.add_argument('-punc', type=int, default=1, help='delete punctuation in text')
    parser.add_argument('-write-train', type=str, default='data/train_stopword_nopunc.csv', help='write train file name')
    parser.add_argument('-write-test', type=str, default='data/test_stopword_nopunc.csv', help='write test file name')
    parser.add_argument('-test', type=int, default=0, help='clean test file')
    parser.add_argument('-train', type=int, default=1, help='clean train file')
    args = parser.parse_args()
    if args.train:
        if args.stop_word:
            stopwords = codecs.open('data/stopwords.txt', encoding='utf-8')
            words = []
            for _ in stopwords:
                words.append(''.join(_.split()))
        else:
            words = []
    #     print(words)
        train = train_label_process()
    #     print(train)
        for i in train.index:
            title = text_process(str(train.loc[i, 'title']), args.punc, args.stop_word, words)
            train.loc[i, 'title'] = title
            content = text_process(str(train.loc[i, 'content']), args.punc, args.stop_word, words)
            train.loc[i, 'content'] = content
        train.to_csv(args.write_train, index=False, encoding='utf-8')
    if args.test:
        test = pd.read_csv('data/test.csv', index_col=False, encoding='utf-8')
        for i in test.index:
            title = text_process(str(test.loc[i, 'title']), args.punc, args.stop_word, words)
            test.loc[i, 'title'] = title
            content = text_process(str(test.loc[i, 'content']), args.punc, args.stop_word, words)
            test.loc[i, 'content'] = content
        test.to_csv(args.write_test, index=False, encoding='utf-8')
