import codecs
import pandas as pd
import numpy as np
import argparse
import jieba
import os

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
            # clear punctuation
            if is_chinese(_):
                temp_text += _
        else:
            # not clear punctuation
            if is_chinese(_) or is_punctuation(_):
                temp_text += _
    text = temp_text
    # clear stop words flag
    if flag: 
        words = list(jieba.cut(temp_text))
        text_temp = ''
        for _ in words:
            if not is_stopword(_, stopwords):
                text_temp += _
        text = text_temp
    return text

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='clean data')
    # read outside parameters
    parser.add_argument('-stop-word', type=int, default=1, help='delete stop words in text')
    parser.add_argument('-punc', type=int, default=1, help='delete punctuation in text')
    parser.add_argument('-write-train', type=str, default='../data/train_clean.csv', help='write train file name')
    parser.add_argument('-write-test', type=str, default='../data/test_clean.csv', help='write test file name')
    parser.add_argument('-test', type=int, default=0, help='clean test file')
    parser.add_argument('-train', type=int, default=1, help='clean train file')
    args = parser.parse_args()
    # clear stop words
    if args.stop_word:
        stopwords = codecs.open('../data/stopwords.txt', encoding='utf-8')
        words = []
        for _ in stopwords:
            words.append(''.join(_.split()))
    else:
        words = []
    # clean train file
    if args.train:
        train = pd.read_csv('../data/train.csv', index_col=False, encoding='utf-8')
        for i in train.index:
            # clean title
            title = text_process(str(train.loc[i, 'title']), args.punc, args.stop_word, words)
            train.loc[i, 'title'] = title
            # clean content
            content = text_process(str(train.loc[i, 'content']), args.punc, args.stop_word, words)
            train.loc[i, 'content'] = content
        # write to new csv
        train.to_csv(args.write_train, index=False, encoding='utf-8')
        print('clean train done')
    # clean test file
    if args.test:
        test = pd.read_csv('../data/test.csv', index_col=False, encoding='utf-8')
        for i in test.index:
            title = text_process(str(test.loc[i, 'title']), args.punc, args.stop_word, words)
            test.loc[i, 'title'] = title
            content = text_process(str(test.loc[i, 'content']), args.punc, args.stop_word, words)
            test.loc[i, 'content'] = content
        test['title'] = test['title'].fillna('无')
        test['content'] = test['content'].fillna('无')
        # write to new csv
        test.to_csv(args.write_test, index=False, encoding='utf-8')
        print('clean test done')
