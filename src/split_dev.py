import pandas as pd
import numpy as np
import random
import csv
import argparse

def write_csv(content, csv_file, flag=0):
    with open(csv_file, 'a') as f:
        writer = csv.writer(f)
        if not flag:
            writer.writerow(content)
        else:
            writer.writerows(content)

def calculate_text_avelen(file):
    with open(file, 'r') as f:
        contents = f.readlines()
    sum = 0
    for i, _ in enumerate(contents):
        if i==0:
            continue
        line = _.split(',')
        a = len(list(line[2]))
        sum += a
    ave = int(sum/(len(contents)-1))
    return ave            

def get_distribution_two(dataframe):
    "二分类"
    cnt_0 = 0
    cnt_1 = 0
    for i in dataframe.index:
        if dataframe.loc[i, 'label'] == 1:
            cnt_0 += 1
        else:
            cnt_1 += 1
     
    return [cnt_0, cnt_1]

def get_distribution(dataframe):
    cnt_0 = 0
    cnt_1 = 0
    cnt_2 = 0
    cnt_3 = 0
    for i in dataframe.index:
        if dataframe.loc[i, 'label'] == 1:
            cnt_0 += 1
        elif dataframe.loc[i, 'label'] == 2:
            cnt_1 += 1
        elif dataframe.loc[i, 'label'] == 3:
            cnt_2 += 1
        else:
            cnt_3 += 1
    return [cnt_0, cnt_1, cnt_2, cnt_3]


def sample_dev(dataframe, dev_num):
    dev_contents = []
    train_contents = []
    for _ in dev_num:
        samples = random.sample(list(dataframe[dataframe['flag']==0].index), _)
#         print(len(dataframe[dataframe['flag']==0]))
#         print(samples)
        for x in samples:
            dataframe.loc[x, 'flag'] = 1
        dev_content = []
        train_content = []
        for index in dataframe.index:
            if index in samples:
                content = [dataframe.loc[index, 'id'], dataframe.loc[index, 'title'], dataframe.loc[index, 'content'], dataframe.loc[index, 'label']]
                dev_content.append(content) # 1份
            else:
                content = [dataframe.loc[index, 'id'], dataframe.loc[index, 'title'], dataframe.loc[index, 'content'], dataframe.loc[index, 'label']]
                train_content.append(content) # 4份
        dev_contents.append(dev_content)
        train_contents.append(train_content)
    return dev_contents, train_contents
        

def split_data_k(k, num):
    dis = []
    ave = int(num / k)
    for _ in range(0, k-1):
        dis.append(ave)
    dis.append(num - ave*(k-1))
    return dis
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='k折交叉划分训练集验证集')
    parser.add_argument('-k', type=int, default=5, help='k fold')
    parser.add_argument('-file', type=str, default='data/train_dev_stopword_nopunc.csv', help='input train dev file')
    args = parser.parse_args()
    train = pd.read_csv(args.file, index_col=False, encoding='utf-8')
    title_flag = list(train['title'].isna())
    content_flag = list(train['content'].isna())
    train['t_flag'] = title_flag
    train['c_flag'] = content_flag
    train_ = train[(train['t_flag']==False)&(train['c_flag']==False)] # delete title&content = none sample
#     print(len(train_))
#     print(calculate_text_avelen(args.file)) # ave content len = 38
    train_['flag'] = np.zeros(len(train_), dtype=int).tolist()
    distribute = get_distribution(train_) 
    print(distribute)
    dev_contents = []
    train_contents = []
    for i, _ in enumerate(distribute):
        dev_num = split_data_k(args.k, _) # [763, 763, 763, 763, 763]
#         print(dev_num) 
        dataframe = train_[train_['label']==i+1]
#         if i == 0:
#             dataframe = train_[train_['label']==1]
#         else:
#             dataframe = train_[train_['label']!=1]
        dev_content, train_content = sample_dev(dataframe, dev_num) # [5*[763 samples]], [5*[763*4 samples]]
        dev_contents.append(dev_content) # [4* [5*[763 samples]] ]
        train_contents.append(train_content)
    dev_data = []
    train_data = []
    for i in range(args.k):
#         dev_data.append(dev_contents[0][i] + dev_contents[1][i])
#         train_data.append(train_contents[0][i] + train_contents[1][i])
        dev_data.append(dev_contents[0][i] + dev_contents[1][i] + dev_contents[2][i] + dev_contents[3][i])
        train_data.append(train_contents[0][i] + train_contents[1][i] + train_contents[2][i] + train_contents[3][i])
    for i, _ in enumerate(dev_data):
        dev_file = 'data/normal/dev' + str(i) + '.csv'
        write_csv(['id', 'title', 'content', 'label'], dev_file)
        random.shuffle(_)
        write_csv(_, dev_file, 1)
    for j, _ in enumerate(train_data):
        train_file = 'data/normal/train' + str(j) + '.csv'
        write_csv(['id', 'title', 'content', 'label'], train_file)
        random.shuffle(_)
        write_csv(_, train_file, 1)

    