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
    

    def get_train_examples(self, data_dir, batch, bertclient, train_file, concat, shuffle=1):
        lines = self._read_csv(data_dir, train_file)
        if shuffle:
            random.shuffle(lines)
            print('shuffle done')
        return self._create_examples(lines, 'train', bertclient, batch, concat)


    def get_dev_examples(self, data_dir, batch, bertclient, dev_file, concat):
        return self._create_examples(self._read_csv(data_dir, dev_file), 'dev', bertclient, batch, concat)


    def get_test_examples(self, data_dir, batch, bertclient, test_file, concat):
        return self._create_examples(self._read_csv(data_dir, test_file), 'test', bertclient, batch, concat)

  

    def _create_examples(self, lines, set_type, bertclient, batch, concat):
        examples = []
        text_examples = []
        title_examples = []
        content_examples = []
        labels = []
        batch_labels = [] 
        for (i, line) in enumerate(lines):
            if concat:
                if line[2]=='':
                    line[2] = line[1] #title
                if line[1] == '':
                    line[1] = line[2] #content
                title_examples.append(line[1])
                content_examples.append(line[2])
            else:
#                 text = line[1] + line[2]
                if line[2]=='':
                    text = line[1] #title
                else:
                    text = line[2] #content
                text_examples.append(text)
                
            if set_type != 'test': 
                label = int(line[3])-1    # 0，1，2，3
                labels.append(label)
        max_len = int(len(lines)/batch)
        for _ in range(0, max_len):
            if concat == 1:#相加
                data_title = bertclient.encode(title_examples[_*batch:(_+1)*batch]) 
                data_content = bertclient.encode(content_examples[_*batch:(_+1)*batch]) 
                data = data_title + data_content # batch * max_seq_len * 768
                print(data.shape)
            elif concat == 2:#横向
                data_title = bertclient.encode(title_examples[_*batch:(_+1)*batch]) 
                data_content = bertclient.encode(content_examples[_*batch:(_+1)*batch]) 
                data = np.concatenate((data_title, data_content), 2) # batch * max_seq_len * 768*2
                print(data.shape)
            elif concat == 3:#竖向
                data_title = bertclient.encode(title_examples[_*batch:(_+1)*batch]) 
                data_content = bertclient.encode(content_examples[_*batch:(_+1)*batch]) 
                data = np.concatenate((data_title, data_content), 1) # batch * max_seq_len*2 * 768
                print(data.shape)
            else:
                data = bertclient.encode(text_examples[_*batch:(_+1)*batch]) 
#                 print(data.shape)
            if set_type != 'test': 
                label = labels[_*batch:(_+1)*batch] # list
                batch_labels.append(label)
#             print(data.shape)
            now = datetime.datetime.now()
            print('['+str(now)+']:'+set_type+' batch '+str(_)+' disposed')
            examples.append(data)
            
        if len(lines) > max_len*batch:
            if concat == 1:#相加
                data_title = bertclient.encode(title_examples[max_len*batch:len(lines)]) 
                data_content = bertclient.encode(content_examples[max_len*batch:len(lines)]) 
                data = data_title + data_content # batch * max_seq_len * 768
                print(data.shape)
            elif concat == 2:#横向
                data_title = bertclient.encode(title_examples[max_len*batch:len(lines)]) 
                data_content = bertclient.encode(content_examples[max_len*batch:len(lines)]) 
                data = np.concatenate((data_title, data_content), 2) # batch * max_seq_len * 768*2
                print(data.shape)
            elif concat == 3:#竖向
                data_title = bertclient.encode(title_examples[max_len*batch:len(lines)]) 
                data_content = bertclient.encode(content_examples[max_len*batch:len(lines)]) 
                data = np.concatenate((data_title, data_content), 1) # batch * max_seq_len*2 * 768
                print(data.shape)
            else:
                data = bertclient.encode(text_examples[max_len*batch:len(lines)]) 
                print(data.shape)
            if set_type != 'test': 
                label = labels[max_len*batch:len(lines)]
                batch_labels.append(label)
#             print(data.shape)
            now = datetime.datetime.now()
            print('['+str(now)+']:'+set_type+' batch '+str(max_len)+' disposed')
            examples.append(data)
        
        if set_type == 'test':
            return examples
        else:
            return examples, batch_labels
        
        
# def write_train_lines_json(train_lines, file):
#     train_list = []
#     train_dict = {}
#     for item in train_lines:
#         temp_item = item[0].tolist()
#         train_dict['data'] = temp_item
#         train_dict['label'] = item[1]
#         train_list.append(train_dict)
#         train_dict = {}
#     with open(file, 'w') as f:
#         f.write(json.dumps(train_list))



# def shuffle_train_lines(train_lines, batch, flag):
#     if flag:
#         random.shuffle(train_lines)
#     batch_labels = [] 
#     examples = []
#     max_len = int(len(train_lines)/batch)
#     for _ in range(0, max_len):
#         data_np = np.array([train_lines[_*batch][0]]) # 1*128*768
#         label = []
# #         label_np = np.array([train_lines[_*batch][1]])
#         for index, d in enumerate(train_lines[_*batch:(_+1)*batch]):
#             if index != 0:
#                 data_np = np.vstack((data_np, np.array([d[0]])))
# #                 label_np = np.vstack((label_np, np.array([d[1]])))
# #         print(data_np.shape)
            
#             label.append(d[1])
# #         print(type(label))
# #         print(label)
#         batch_labels.append(label)
# #             now = datetime.datetime.now()
# #             print('['+str(now)+']:'+set_type+' batch '+str(_)+' disposed')
#         examples.append(data_np) # batch*128*768

#     if len(train_lines) > max_len*batch:
# #         if len(train_lines)-max_len*batch == 1:
#         data_np = np.array([train_lines[max_len*batch][0]]) # 1*128*768
# #         else:
# #             data_np = train_lines[max_len*batch][0] 
# #         label_np = np.array([train_lines[max_len*batch][1]])
#         label = []
#         for index, d in enumerate(train_lines[max_len*batch:len(train_lines)]):
#             if index != 0:
#                 data_np = np.vstack((data_np, np.array([d[0]])))
# #                 label_np = np.vstack((label_np, np.array([d[1]])))
#             label.append(d[1])
# #         print(data_np.shape)
#         batch_labels.append(label)
# #             now = datetime.datetime.now()
# #             print('['+str(now)+']:'+set_type+' batch '+str(max_len)+' disposed')
#         examples.append(data_np)
# #     print(len(examples), len(batch_labels))
#     return examples, batch_labels




        