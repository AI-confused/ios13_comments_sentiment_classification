import pandas as pd
import csv
import argparse
import numpy as np

def write_csv(content, csv_file, flag=0):
    with open(csv_file, 'a') as f:
        writer = csv.writer(f)
        if not flag:
            writer.writerow(content)
        else:
            writer.writerows(content)
        
        
def read_csv(file):
    with open(file, 'r') as f:
        reader = csv.reader(f)
        logit = np.array(list(reader), dtype=float)
        return logit # 7356 * 3
        
        

def result_process(file, write_file):
    dataframe = pd.read_csv(file, header=None, names=['0', '1', '2'], index_col=False, encoding='utf-8')
    id_frame = pd.read_csv('../data/test.csv', index_col=False, encoding='utf-8')
    write_csv(['id', 'label'], write_file)
    labels = []
    for i in dataframe.index:
        max_value = dataframe.loc[i, '0']
        max_index = 0
        if dataframe.loc[i, '1'] >= max_value:
            max_value = dataframe.loc[i, '1']
            max_index = 1
        if dataframe.loc[i, '2'] >= max_value:
            max_value = dataframe.loc[i, '2']
            max_index = 2
        content = [id_frame.loc[i, 'id'], max_index]
        write_csv(content, write_file)

    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='choose test result path')
    parser.add_argument('-test', type=str, default='output/test_result.csv')
    parser.add_argument('-output', type=str, default='output/final.csv')
    parser.add_argument('-data', type=str, default='output/')
    parser.add_argument('-fold', type=int, default=1)
    args = parser.parse_args()
    if args.fold:
        files = ['result0.csv', 'result1.csv', 'result2.csv', 'result3.csv', 'result4.csv']
        logits = np.zeros((7356, 3), dtype=float)
        for file in files:
            address = args.data + file
            logits += read_csv(address)
        logits = logits / 5
        logits = logits.tolist()
        write_csv(logits, args.test, 1)
    result_process(args.test, args.output)
        



