import os
import argparse
import datetime
import torch
from bert_serving.client import BertClient
import json
import csv
import random
import numpy as np


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir, bertclient, batch):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir, bertclient, batch):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir, bertclient, batch):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()
    
    
    
class MyTaskProcessor(DataProcessor):
    def _read_csv(self, data_dir, file_name):
        with open(data_dir+'/'+file_name, 'r') as f:
            reader = csv.reader(f, delimiter=',', quotechar=None)
            lines = []
            for i, line in enumerate(reader):
                if i != 0:
                    lines.append(line)
        return lines
    
    def get_labels(self):
        return ['1', '2', '3', '4']
    

    def get_train_examples(self, data_dir, batch, bertclient, train_file, shuffle=1):
        lines = self._read_csv(data_dir, train_file)
        if shuffle:
            random.shuffle(lines)
            print('shuffle done')
        return self._create_examples(lines, 'train', bertclient, batch)


    def get_dev_examples(self, data_dir, batch, bertclient, dev_file):
        return self._create_examples(self._read_csv(data_dir, dev_file), 'dev', bertclient, batch)


    def get_test_examples(self, data_dir, batch, bertclient, test_file):
        return self._create_examples(self._read_csv(data_dir, test_file), 'test', bertclient, batch)

  

    def _create_examples(self, lines, set_type, bertclient, batch):
        examples = []
        text_examples = []
        title_examples = []
        content_examples = []
        labels = []
        batch_labels = [] 
        for (i, line) in enumerate(lines):  
            text = line[3] #content
            if text=='':
                text = '无'
            text_examples.append(text)
            if set_type != 'test': 
                label = int(line[4])-1    # 0，1，2，3
                labels.append(label)
        max_len = int(len(lines)/batch)
        for _ in range(0, max_len):
            data = bertclient.encode(text_examples[_*batch:(_+1)*batch]) 
            if set_type != 'test': 
                label = labels[_*batch:(_+1)*batch] # list
                batch_labels.append(label)
            now = datetime.datetime.now()
            print('['+str(now)+']:'+set_type+' batch '+str(_)+' disposed')
            examples.append(data)
            
        if len(lines) > max_len*batch:
            data = bertclient.encode(text_examples[max_len*batch:len(lines)]) 
            if set_type != 'test': 
                label = labels[max_len*batch:len(lines)]
                batch_labels.append(label)
            now = datetime.datetime.now()
            print('['+str(now)+']:'+set_type+' batch '+str(max_len)+' disposed')
            examples.append(data)
        
        if set_type == 'test':
            return examples
        else:
            return examples, batch_labels




        
