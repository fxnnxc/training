

echo '------------------------------------------'
echo '---------------- TRAIN ALL ---------------'
echo 'data_path'      : $data_path 
echo 'data_path'      : $data_path 
echo 'model'          : $model 
echo "JihyeonCNNClassifier / JihyeonLSTMClassifier"
echo '------------------------------------------'

# ------------ CIFAR10 ALL ------------ 
\

hidden_dim=128 
python scripts/jihyeon_classification.py \
        --data-path $data_path \
        --epochs 10 \
        --lr 1e-1 \
        --root hub \
        --eval-freq 1 \
        --optim-type sgd \
        --scheduler-type cosine \
        --batch-size 32 \
        --model $model \
        --hidden-dim $hidden_dim
