python train.py --model resnet34unet --learning_rate 0.012 --epoch 60  --optim sgd --split 0 --image_size 1024 -b 8 --dr 0 --snapshot 1
python train.py --model resnet34unet --learning_rate 1e-4 --epoch 20  --optim sgd --split 0 --lovasz True --resume True --image_size 1024 -b 8 --dr 0 --snapshot 1
