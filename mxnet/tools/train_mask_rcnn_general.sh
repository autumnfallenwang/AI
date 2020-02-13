BUILD=python
SCRIPT=train_mask_rcnn_general.py

$BUILD $SCRIPT \
--gpus 0,1,2,3,4,5,6,7 \
-j 8 \
--network resnet50_v1b \
--dataset coco
#--lr 0.001 \
#--lr-decay-epoch 160,200 \
#--lr-decay 0.1 \
#--epochs 240


