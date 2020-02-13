BUILD=python
CONVERT=/home/wangqiushi/anaconda3/envs/mxnet/lib/python3.7/site-packages/mxnet/tools/im2rec.py
LOCATION=/raid/data/kaggle/diabetic_retinopathy/mxnet_rec

PREFIX=$LOCATION/rec
THREAD=16

echo "Create train rec.."
rm -rf $PREFIX/train*
$BUILD $CONVERT \
$PREFIX/train \
$LOCATION/train_folder \
--list \
--recursive \

$BUILD $CONVERT \
$PREFIX/train \
$LOCATION/train_folder \
--pass-through \
--num-thread $THREAD \
--pack-label


echo "Create valid rec.."
rm -rf $PREFIX/valid*
$BUILD $CONVERT \
$PREFIX/valid \
$LOCATION/valid_folder \
--list \
--recursive \
--no-shuffle

$BUILD $CONVERT \
$PREFIX/valid \
$LOCATION/valid_folder \
--pass-through \
--num-thread $THREAD \
--pack-label


echo "Create test rec.."
rm -rf $PREFIX/test*
$BUILD $CONVERT \
$PREFIX/test \
$LOCATION/test_folder \
--list \
--recursive \
--no-shuffle

$BUILD $CONVERT \
$PREFIX/test \
$LOCATION/test_folder \
--pass-through \
--num-thread $THREAD \
--pack-label


echo "All Done.."
