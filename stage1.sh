#!/usr/bin/env bash

# python train.py --model resnet34unet --learning_rate 5e-3 --epoch 40  --optim sgd --split 0 --image_size 768 --resume True --snapshot 1 --min_lr 5e-4
python train.py --model resnet34unet --learning_rate 5e-4 --epoch 20  --optim sgd --split 0 --lovasz True --resume True --image_size 768 --min_lr 5e-4

python train.py --model resnet34unet --learning_rate 5e-3 --epoch 40  --optim sgd --split 1 --image_size 768  --batch_size 16 --min_lr 5e-4 --snapshot 1
python train.py --model resnet34unet --learning_rate 5e-4 --epoch 20  --optim sgd --split 1 --lovasz True --resume True --image_size 768  --batch_size 16  --min_lr 5e-5
python validate.py --model resnet34unet --split 1 --image_size 768

python train.py --model resnet34unet --learning_rate 5e-3 --epoch 40 --optim sgd --split 2 --image_size 768  --batch_size 16 --min_lr 5e-4 --snapshot 1
python train.py --model resnet34unet --learning_rate 5e-4 --epoch 20 --optim sgd --split 2 --lovasz True --resume True --image_size 768  --batch_size 16 --min_lr 5e-5
python validate.py --model resnet34unet --split 2 --image_size 768

python train.py --model resnet34unet --learning_rate 5e-3 --epoch 40 --optim sgd --split 3 --image_size 768  --batch_size 16 --min_lr 5e-4 --snapshot 1
python train.py --model resnet34unet --learning_rate 5e-4 --epoch 20 --optim sgd --split 3 --lovasz True --resume True --image_size 768  --batch_size 16 --min_lr 5e-5
python validate.py --model resnet34unet --split 3 --image_size 768

python train.py --model resnet34unet --learning_rate 5e-3 --epoch 40  --optim sgd --split 4 --image_size 768  --batch_size 16 --min_lr 5e-4 --snapshot 1
python train.py --model resnet34unet --learning_rate 5e-4 --epoch 20  --optim sgd --split 4 --lovasz True --resume True --image_size 768  --batch_size 16 --min_lr 5e-4
python validate.py --model resnet34unet --split 4 --image_size 768


