python train.py --model resnet34unet --learning_rate 0.005 --min_lr 5e-4 --epoch 40  --optim sgd --split 0 --image_size 768 -b 16 --use_total True --snapshot 1
python train.py --model resnet34unet --learning_rate 5e-4 --epoch 20  --optim sgd --split 0 --lovasz True --resume True --image_size 768 --min_lr 1e-4 --use_total True -b 16 --snapshot 3