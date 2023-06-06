

echo '------------------------------------------'
echo '---------------- TRAIN ALL ---------------'
echo 'data_path'      : $data_path 
echo 'data_path'      : $data_path 
echo '------------------------------------------'

# ------------ CIFAR10 ALL ------------ 

models="
JihyeonCNNClassifier
JihyeonLSTMClassifier
"
for model in $models 
do 
        for hidden_dim in 32 64 128 
        do 
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
        done 
done 