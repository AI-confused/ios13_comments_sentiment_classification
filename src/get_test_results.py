import pandas as pd
import csv
import argparse
import numpy as np


   

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='choose test result path')
    parser.add_argument("-output", default=None, type=str, required=True)
    parser.add_argument("-k", default=None, type=int, required=True)
    args = parser.parse_args()

    df=pd.read_csv('../data/test_clean.csv')
    df['1']=0
    df['2']=0
    df['3']=0
    df['4']=0
    for i in range(args.k):
        temp=pd.read_csv('../output/model_textcnn/fold_{}/test_result_{}.csv'.format(args.k,i))
        df['1']+=temp['label_1']/args.k
        df['2']+=temp['label_2']/args.k
        df['3']+=temp['label_3']/args.k
        df['4']+=temp['label_3']/args.k
    print(df['1'].mean())

    df['label']=np.argmax(df[['1','2','3', '4']].values,-1)+1 # 1,2,3,4
    df[['id','label']].to_csv(args.output,index=False)
    nums = [n1=0, n2=0, n3=0, n4=0]
    for _ in df.index:
        if df.loc[_, 'label'] == 1:
            nums[n1] += 1
        elif df.loc[_, 'label'] == 2:
            nums[n2] += 1
        elif df.loc[_, 'label'] == 3:
            nums[n3] += 1
        else:
            nums[n4] += 1
    print(nums)


        



