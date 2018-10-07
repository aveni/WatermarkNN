# Turning Your Weakness Into a Strength: Watermarking Deep Neural Networks by Backdooring
Deep Neural Networks have recently gained lots of success after enabling several breakthroughs in notoriously challenging problems. Training these networks is computationally expensive and requires vast amounts of training data. Selling such pre-trained models can, therefore, be a lucrative business model. Unfortunately, once the models are sold they can be easily copied and redistributed. To avoid this, a tracking mechanism to identify models as the intellectual property of a particular vendor is necessary. 
In this work, we present an approach for watermarking Deep Neural Networks in a black-box way. Our scheme works for general classification tasks and can easily be combined with current learning algorithms. We show experimentally that such a watermark has no noticeable impact on the primary task that the model is designed for and evaluate the robustness of our proposal against a multitude of practical attacks. Moreover, we provide a theoretical analysis, relating our approach to previous work on backdooring.

If you find our work useful please cite: 
[Turning Your Weakness Into a Strength: Watermarking Deep Neural Networks by Backdooring] (https://www.usenix.org/system/files/conference/usenixsecurity18/sec18-adi.pdf)
```
@inproceedings {217591,
author = {Yossi Adi and Carsten Baum and Moustapha Cisse and Benny Pinkas and Joseph Keshet},
title = {Turning Your Weakness Into a Strength: Watermarking Deep Neural Networks by Backdooring},
booktitle = {27th {USENIX} Security Symposium ({USENIX} Security 18)},
year = {2018},
isbn = {978-1-931971-46-1},
address = {Baltimore, MD},
pages = {1615--1631},
url = {https://www.usenix.org/conference/usenixsecurity18/presentation/adi},
publisher = {{USENIX} Association},
}
```

## Content
The repository contains code for training and fine-tunning watermarked neural network models. It contains three main scripts: `train.py, predict.py, and fine-tune.py` where you can train, predict and fine-tune these models. 

Additionally, the repo contains the trigger set images used to embed the watermark.

At the moment the code only supporst training and evaluating on [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset. Other datasets will be supported soon. 

## Dependencies
[Python3.6](https://www.anaconda.com/download)
[PyTorch0.4.1](https://pytorch.org/)

## Usage
The `train.py` script allows you to train a model with or without a trigger set. 

### Example
For training with the trigger set:
```
python main.py --batch_size 100 --max_epochs 60 --runname train --wm_batch_size 2 --wmtrain
```
For training without the trigger set, just omit the --wmtrain flag.
