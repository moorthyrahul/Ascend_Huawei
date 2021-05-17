
<!-- Insert header and hyperlinks to documentations and Ascend community, and ModelZoo-->
[![Header](https://r.huaweistatic.com/s/ascendstatic/lst/header/header-logo.png "Header")](https://www.hiascend.com/software/modelzoo)

![](https://img.shields.io/badge/Ascend-910-informational?style=flat&logo=huawei&logoColor=white&labelColor=080404&color=c31d20)
![](https://img.shields.io/badge/OS-Linux-informational?style=flat&logo=linux&logoColor=white&labelColor=080404&color=c31d20)
![](https://img.shields.io/badge/Code-Python-informational?style=flat&logo=python&logoColor=white&labelColor=080404&color=c31d20)
# Model Training on Huawei's Ascend Environment
 
In this blog we discuss about training deep learning models on Huawei's Ascend environment using Ascend 910 processor. We tried to experiment with various hyperparameters and demonstrate better training performance and loss convergence. For better understanding the environment, we selected to train the AlexNet model for image classification task.

## Introduction to AlexNet Training
<hr/>
AlexNet is a classic image classification network that won the 2012 ImageNet competition. It is the first CNN that successfully applies tricks such as ReLU, Dropout, and LRN, and uses GPUs to accelerate computation. The AlexNet used is based on Tensorflow.

Reference: https://arxiv.org/abs/1404.5997 

Implementation :  https://www.hiascend.com/en/software/modelzoo/detail/2/6b29663a9ab64fde8ad8b9b2ddaf2b05


## Default Configuration for Training
<hr/>
<!--Following sections introduce the default configurations and hyperparameters for AlexNet model.
Optimizer: -->
This model uses the Momentum optimizer from Tensorflow with the following default configurations and hyperparameters:

| Hyperparameters    | Value   |
| -------------------|---------| 
|  Momentum          |   0.9   |
|  Learning rate (LR)|   0.06  |
|  LR schedule       |   cosine annealing  |
| Batch size | 128*8 for 8 NPUs, 256 for single NPU | 
| Weight decay       | 0.0001 | 
| Label smoothing    |  0.1   |

Below are brief descriptions of each hyperparameter:
<!-- Momentum: 0.9<br/>
Learning rate (LR) : 0.06<br/>
LR schedule: cosine_annealing<br>
Batch size : 128*8 for 8 NPUs, 256 for single NPU <br/>
Weight decay : 0.0001. <br/>
Label smoothing = 0.1
-->
- **Momentum** : helps accelerate gradient descent in the relevant direction by dampening oscillator, thus leading to faster convergence.<br/>
- **Learning Rate** : float value that determines the step size of gradient descent when optimizing toward the optimal.
- **LR schedule** : adjusts the learning rate during optimizing.

## Training dataset preprocessing
<hr/>
The input image size for this model is 224 x 224 and are preprocessed before sending them to the model for training. The preprocessing step include:

- Random crop 
- Horizontal flip
- Normalization

Note: You may implement your own preprocessing technique in `preprocessing.py`


## Experiments: 
<hr/>

We implemented the following parameter changes to give our observations on changes in batch time, loss convergence and training time:

1. **inter_op_parallelism_threads & inter_op_parallelism_threads (CreateSession.py)**: Used by tensorflow to The execution of an individual op (for some op types) can be parallelized on a pool of intra_op_parallelism_threads. 0 means the system picks an appropriate number.
2. **allow_soft_placement (CreateSession.py)**: If this option is enabled (=True), the operation will be be placed on CPU if there:

   i. No GPU devices are registered or known
   
   ii. No GPU implementation for the operation
   
   This option only works when your tensorflow is not GPU compiled. If your tensorflow is GPU supported, no matter if allow_soft_placement is set or not and even if you set          device as CPU.

   | Precision Mode |  Mode | Avg Batch Time  |
   | ---------------|---------------|-------------|
   |  `allow_soft_placement`    | True |  ~5.4s  |
   |  `allow_soft_placement` |  False |  ~5.5s   |
   
3. **xla**:
4. **precision_mode (trainer.py)**: Mixed precision is the combined use of the float16 and float32 data types in training deep neural networks, which reduces memory usage and access frequency. Mixed precision training makes it easier to deploy larger networks without compromising the network accuracy with float32.

    - **allow_mix_precision**: Mixed precision is allowed to improve system performance and reduce memory usage with little accuracy loss.
    - **must_keep_origin_dtype**: Retains original precision. 
    - **allow_fp32_to_fp16**: The original precision is preferentially retained. If an operator does not support the float32 data type, the float16 precision is used. 
    - **force_fp16**: If an operator supports both float16 and float32 data types, float16 is forcibly selected.
    
    **Results:**
    The following table compares the loss, accuracy and batch time obtained by using the four precision mode with the baseline. We see that setting `allow_mix_precision=True`         yields the best performace in this experiment setting. 

    | Precision Mode | Loss/Accuracy | Batch Time  |
    | ---------------|---------------|-------------|
    |  `allow_mix_precision`    |   = Baseline   | ~50ms  |
    |  `must_keep_origin_dtype` |   N/A          | NA     |
    |  `allow_fp32_to_fp16`     |   = Baseline   | ~170ms |
    |  `force_fp16`             |   < Baseline   | ~50ms  |

The figure below shows the Top1 accuracy curve under different precision mode:

<!-- <img align="center" src="./assets/experiment_results_1.png"> -->

![alt text](./assets/experiment_results_1.png )
 
Top1 accuracy curse:

•	**Blue:** allow_fp32_to_fp16


•	**Green:** allow_mix_precision


•	**Purple:** force_fp16

Note, using ‘must_keep_origin_dtype’ results in Error:

![alt text](./assets/keep_origin_dtype.png )

5. **hcom_parallel (trainer.py):**

Whether to enable the AllReduce gradient update and forward and backward parallel execution.

•	**True:** enabled

•	**False (default):** disabled


### Results
Tested on one NPU, no difference in either loss or batch time


## Project Layout
<hr/>
Include directory structure here (tree command)

TODO: reformat experiment, parametes from Derek's doc
