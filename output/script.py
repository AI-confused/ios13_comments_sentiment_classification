import os

for _ in range(5):
    os.system('cp ~/lyl/CCF_classfy/CCF-BDCI-Sentiment-Analysis-Baseline/apple_model0/apple_model_bert_'+str(_)+'/eval_results.txt ./roberta_large/eval_results_'+str(_)+'.txt')
