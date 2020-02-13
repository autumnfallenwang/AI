BUILD=python
SCRIPT=train_ssd_general.py

$BUILD $SCRIPT \
--gpus 0 \
-j 8 \
--network vgg16_atrous \
--data-shape 300 \
--batch-size 16 \
--dataset voc \
--lr 0.001 \
--lr-decay-epoch 160,200 \
--lr-decay 0.1 \
--epochs 240


