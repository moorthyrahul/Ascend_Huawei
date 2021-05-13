
# Model Training on Huawei's Ascend Environment
 
In this blog we discuss about training deep learning models on Huawei's Ascend environment using Ascend 910 processor. We tried to experiment with various hyperparameters and demonstrate better training performance and loss convergence. For better understanding the environment, we selected to train the AlexNet model for image classification task.

## Introduction to AlexNet Training
AlexNet is a classic image classification network that won the 2012 ImageNet competition. It is the first CNN that successfully applies tricks such as ReLU, Dropout, and LRN, and uses GPUs to accelerate computation. The AlexNet used is based on Tensorflow.

Reference: https://arxiv.org/abs/1404.5997 

Implementation :  https://www.hiascend.com/en/software/modelzoo/detail/2/6b29663a9ab64fde8ad8b9b2ddaf2b05


## Default Configuration for Training
Following sections introduce the default configurations and hyperparameters for AlexNet model.
Optimizer: This model uses Momentum optimizer from Tensorflow with the following hyperparameters:

Momentum : 0.9
Learning rate (LR) : 0.06
LR schedule: cosine_annealing
Batch size : 128*8 for 8 NPUs, 256 for single NPU 
Weight decay : 0.0001. 
Label smoothing = 0.1
Momentum : Helps accelerate gradient descent in the relevant direction by dampening oscillator, thus leading to faster convergence.
Learning Rate : Float value that determines the step size of gradient descent when optimizing toward the optimal.
LR schedule : Scheduler that adjusts the learning rate during optimizing.

## Training dataset preprocessing:
Input image size: 224 x 224. Randomly crop the image, flip image horizontally. Normalize each input image by using the mean and standard deviation.

## Experiments: 

We implemented the following parameter changes to give our observations on changes in batch time, loss convergence and training time:

1. **inter_op_parallelism_threads & inter_op_parallelism_threads (CreateSession.py)**: The execution of an individual op (for some op types) can be parallelized on a pool of intra_op_parallelism_threads. 0 means the system picks an appropriate number.

2. **precision_mode (trainer.py)**: Mixed precision is the combined use of the float16 and float32 data types in training deep neural networks, which reduces memory usage and access frequency. Mixed precision training makes it easier to deploy larger networks without compromising the network accuracy with float32.

3. **allow_fp32_to_fp16**: The original precision is preferentially retained. If an operator does not support the float32 data type, the float16 precision is used. Currently, the float32 type is not supported by convolution operators, such as Conv2D and DepthwiseConv2D. These operators are precision-insensitive and do not reduce the accuracy of the entire network.
