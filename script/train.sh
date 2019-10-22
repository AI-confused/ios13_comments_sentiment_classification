k=10
mkdir ../output/model_textcnn
mkdir ../output/model_textcnn/fold_$k
cd ../src
for((i=7;i<k;i++));  
do   
python3 main.py \
-server-ip='10.15.82.239' \
-predict=1 \
-training=1 \
-port=8190 \
-port-out=5556 \
-model=../output/model_textcnn/fold_$k/model_$i.pkl \
-train-file=fold_$k/data_$i/train.csv \
-dev-file=fold_$k/data_$i/dev.csv \
-test-file=fold_$k/data_$i/test.csv \
-predict-file=../output/model_textcnn/fold_$k/test_result_$i.csv \
-load-model=../output/model_textcnn/fold_$k/model_$i.pkl \
-eval-result=../output/model_textcnn/fold_$k/eval_result_$i.txt

done
