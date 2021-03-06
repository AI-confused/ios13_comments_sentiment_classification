import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import data
import datetime
from torch.autograd import Variable
import numpy as np
import csv

class ConfMatrix(object):
    def __init__(self, num_label):
        self.conf = np.zeros((num_label,num_label), dtype=int)
        
    def change_conf(self, i, j):
        self.conf[i][j] += 1
          
    def get_macro_f1(self, label, num_label):
        eplison = 1e-8
        pre_sum = 0
        recall_sum = 0
        for _ in range(num_label):
            pre_sum += self.conf[_][label]
            recall_sum += self.conf[label][_]
        prediction = float(self.conf[label][label]/(pre_sum+eplison))
        recall = float(self.conf[label][label]/(recall_sum+eplison))
        macro_f1 = float((2*prediction*recall) / (prediction + recall + eplison))
        return macro_f1
        
    def get_average_macro_f1(self, macro_f1_list):
        sum = 0
        for _ in macro_f1_list:
            sum += _
        return float(sum/len(macro_f1_list))
    
    

def train(model, train, train_label, loss_func, optimizer, epoch, batch, device, eval_result):
    model.train()
    accu_loss = 0
    for _ in range(len(train)):
        output = model(Variable(torch.FloatTensor(train[_])).to(device))# batch * 4
        label = np.array(train_label[_])
        label = Variable(torch.LongTensor(label)).to(device)
        loss = loss_func(output, label)
        accu_loss += loss.mean().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    now = datetime.datetime.now()
    with open(eval_result, 'a') as f:
        f.write('['+str(now)+']:epoch '+str(epoch)+' loss: '+str(accu_loss/len(train))+'\n')
    print('['+str(now)+']:epoch '+str(epoch)+' loss: '+str(accu_loss/len(train)))


def eval(model, dev, dev_label, batch, epoch, device, num_label, eval_result):
    model.eval()
    right_count = 0
    confmatrix = ConfMatrix(num_label)
    for _ in range(len(dev)):
        output = model.forward(Variable(torch.FloatTensor(dev[_])).to(device)).detach()
        output = output.max(1)[1]
        for y, label in enumerate(dev_label[_]):
            if output.cpu().numpy()[y] == label:
                right_count += 1
            confmatrix.change_conf(label, output.cpu().numpy()[y])
    accuracy = float(right_count/(batch*(len(dev)-1)+len(dev[-1])))
    macro_f1_list = []
    for j in range(num_label):
        macro_f1_list.append(confmatrix.get_macro_f1(j, num_label))
    macro_f1 = confmatrix.get_average_macro_f1(macro_f1_list)
    now = datetime.datetime.now()
    with open(eval_result, 'a') as f:
        f.write('['+str(now)+']:epoch '+str(epoch)+': accuracy: '+str(accuracy) + '| macro f1: ' + str(macro_f1)+'\n')
        f.write('*'*30+'\n')
    print('['+str(now)+']:epoch '+str(epoch)+': accuracy: '+str(accuracy) + '| macro f1: ' + str(macro_f1))
    return macro_f1

def write_csv(content, csv_file):
    with open(csv_file, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(content)
    
    
    
def predict(model, test, device, file):
    model.eval()
    write_csv(['label_1', 'label_2', 'label_3' ,'label_4'], file)
    for _ in range(len(test)):
        output = model.forward(Variable(torch.FloatTensor(test[_])).to(device)).detach()
        output = output.cpu().numpy().tolist()
        for item in output:
            write_csv(item, file)



