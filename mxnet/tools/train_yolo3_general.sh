BUILD=python
SCRIPT=train_yolo3_general.py

$BUILD $SCRIPT \
--network darknet53 \
--dataset voc \
--gpus 0,1,2,3,4,5,6,7 \
--batch-size 64 \
-j 8 \
--log-interval 100 \
--lr-decay-epoch 160,180 \
--epochs 200 \
--syncbn \
--warmup-epochs 4


