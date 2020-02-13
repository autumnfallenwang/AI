BUILD=python
CONVERT=/home/wangqiushi/.local/lib/python3.6/site-packages/mxnet/tools/im2rec.py
LOCATION=/data02/wangqiushi/datasets/DR

PREFIX=$LOCATION/mxnet_rec/t90
THREAD=16

echo "Create train rec.."
rm -rf $PREFIX/train.rec
rm -rf $PREFIX/train.idx
$BUILD $CONVERT \
$PREFIX/train \
$LOCATION/Images \
--pass-through \
--num-thread $THREAD \
--pack-label


echo "Create valid rec.."
rm -rf $PREFIX/valid.rec
rm -rf $PREFIX/valid.idx
$BUILD $CONVERT \
$PREFIX/valid \
$LOCATION/Images \
--pass-through \
--num-thread $THREAD \
--pack-label


echo "All Done.."
