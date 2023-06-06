

echo '------------------------------------------'
echo '---------------- TRAIN ALL ---------------'
echo 'jihyeon_path'      : $jihyeon_path 
echo 'jihyeon_path'      : $jihyeon_path 
echo '------------------------------------------'

# ------------ CIFAR10 ALL ------------ 

models="
JihyeonCNNClassifier
JihyeonLSTMClassifier
"
for model 
python scripts/jihyeon_classification.py \
        --data-path $jihyeon_path \
        --epochs 10 \
        --lr 1e-1 \
        --root hub \
        --eval-freq 1 \
        --optim-type sgd \
        --scheduler-type cosine \
        --batch-size 32


