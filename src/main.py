import os
import argparse
import datetime
import torch
from bert_serving.client import BertClient
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import csv
import json
import data
import model
import train


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN text classificer')
    # learning
    parser.add_argument('-lr', type=float, default=1e-3, help='initial learning rate [default: 0.001]')
    parser.add_argument('-epoch', type=int, default=20, help='number of epochs for train [default: 20]')
    parser.add_argument('-batch', type=int, default=512, help='batch size for training [default: 512]')
    parser.add_argument('-save', type=int, default=1, help='whether to save model')
    parser.add_argument('-model', type=str, default='output/text_cnn.pkl', help='dir to store model')
    parser.add_argument('-predict-file', type=str, default='output/test_result.csv', help='dir to store predict')
    # data 
    parser.add_argument('-data-dir', type=str, default='data', help='dataset dir')
    parser.add_argument('-train-file', type=str, default='train0.csv', help='train file')
    parser.add_argument('-dev-file', type=str, default='dev0.csv', help='dev file')
    parser.add_argument('-test-file', type=str, default='test_stopword_nopunc.csv', help='test file')
    # model
    parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
    parser.add_argument('-kernel-num', type=int, default=768, help='number of each kind of kernel')
    parser.add_argument('-concat', type=int, default=0, help='type of dispose with data')
    parser.add_argument('-kernel', type=int, default=3, help='kernel size')
    # device
    parser.add_argument('-remote', type=int, default=0, help='use local or remote device with bert-as-service')
    parser.add_argument('-predict', type=int, default=0, help='predict the sentence given')
    parser.add_argument('-static-epoch', type=int, default=0, help='static epoch num or dynamic epoch num')
    parser.add_argument('-training', type=int, default=1, help='train or test')
    parser.add_argument('-load-model', type=str, default='output/text_cnn.pkl', help='predict loading model dir')
    parser.add_argument('-port', type=int, default=8190, help='bert-as-service port')
    parser.add_argument('-port-out', type=int, default=5556, help='bert-as-service port')
    args = parser.parse_args()

    if args.remote:
        bc = BertClient(ip='10.15.82.239',port=args.port, port_out=args.port_out, check_version=False)
    else:
        bc = BertClient(port=8190, check_version=False)
    mydata = data.MyTaskProcessor()
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    print('device: '+str(device))    
    # update args and print
    args.class_num = len(mydata.get_labels()) # 4
    # define text cnn model
    if args.kernel == 2:
        kernel_size = [2]
    elif args.kernel == 3:
        kernel_size = [2, 3]
    elif args.kernel == 4:
        kernel_size = [2, 3, 4]
    cnn = model.CNN_Text(1, args.kernel_num, kernel_size, args.class_num, args.dropout, args.concat).to(device) #need to modify
  
    if args.training:
        dev, dev_label = mydata.get_dev_examples(args.data_dir, args.batch, bc, args.dev_file, args.concat)
        print('start training')  
        loss_func = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1.05740498, 1, 1.10068213, 1.40410721])).float(), size_average=True)
        loss_func.to(device)
        optimizer = torch.optim.Adam(cnn.parameters(), lr=args.lr)   
        max_f1 = 0
        middle_f1 = 0
        
        for i in range(args.epoch):
            train_, train_label = mydata.get_train_examples(args.data_dir, args.batch, bc, args.train_file, args.concat)
            train.train(cnn, train_, train_label, loss_func, optimizer, i, args.batch, device)
            macro_f1 = train.eval(cnn, dev, dev_label, args.batch, i, device, args.class_num)
            
            if not args.static_epoch:
                if macro_f1 >= max_f1:
                    max_f1 = macro_f1
                elif macro_f1 >= middle_f1 and macro_f1 < max_f1:
#                     stop = int(input('stop training?'))
#                     if stop:
#                         break
                    middle_f1 = macro_f1
                else: 
#                     stop = int(input('stop training?'))
#                     if stop:
                    break
        print('max macro f1:' + str(max_f1))


        if args.save == 1:
            torch.save(cnn.state_dict(), args.model) #保存网络参数
            print('model saved')
    if args.predict:
        cnn.load_state_dict(torch.load(args.load_model))
        print('model load done')
        test = mydata.get_test_examples(args.data_dir, args.batch, bc, args.test_file, args.concat)
        print('test data done')
        train.predict(cnn, test, device, args.predict_file)
        print('predict saved')
    bc.close()


    
    
    
    