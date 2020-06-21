# Temporal graph-rcnn for self-driving systems
Pytorch code for our master's 2020 capstone project which is an improvement on ECCV 2018 paper ["Graph R-CNN for Scene Graph Generation"](https://arxiv.org/pdf/1808.00191.pdf)

<!-- :balloon: 2019-06-04: Okaaay, time to reimplement Graph R-CNN on pytorch 1.0 and release a new benchmark for scene graph generation. It will also integrate other models like IMP, MSDN and Neural Motif Network. Stay tuned!

:balloon: 2019-06-16: Plan is a bit delayed by ICCV rebuttal, but still on track. Stay tuned! -->

## Introduction
Scene graph generation has shown promising gains over traditional image based deep learning approches on the task of scene understanding and contextual question answering based on images.

However, the orginal implementation is not a good fit for application in self-driving/autonomous driving systems since the existing approach does not take into consideration any temporal features from the sequence of inputs.

We propose a modified and improved architecture which leverages the outstanding work done in the field of scene graph generation and add a neural transformer to the flow to model temporal features which efficiently capture the key behavorial aspects of self-driving cars.

## Why we need this repository?

The goal of this repository is to provide a working implementation of the approach described above. The dataset used is proprietary and a property of Honda Research Institute, USA. Hence, we are unable to publish the dat preprocessing and pipeline at the moment. However, you are more than welcome to train the models on any other suitable dataset.

## Tips and Tricks

Some important observations based on the experiments:

* **Using per-category NMS is important!!!!**. We have found that the main reason for the huge gap between the imp-style models and motif-style models is that the later used the per-category nms before sending the graph into the scene graph generator. Will put the quantitative comparison here.

* **Different calculations for frequency prior result in differnt results***. Even change a little bit to the calculation fo frequency prior, the performance of scene graph generation model vary much. In neural motiftnet, we found they turn on filter_non_overlap, filter_empty_rels to filter some triplets and images.

## Installation

### Prerequisites

* Python 3.6+
* Pytorch 1.0
* CUDA 8.0+

### Dependencies

Install all the python dependencies using pip:
```
pip install -r requirements.txt
```

and libraries using apt-get:
```
apt-get update
apt-get install libglib2.0-0
apt-get install libsm6
```

### Dataset Used

* [Honda Driving Dataset](https://usa.honda-ri.com/hdd):


### Compilation

Compile the cuda dependencies using the following commands:
```
cd lib/scene_parser/rcnn
python setup.py build develop
```

## Train

### Train object detection model:

### Train scene graph generation model jointly (train detector, neural transformer and sgg as a whole):

* Vanilla scene graph generation model with resnet-101 as backbone:
```
python main.py --config-file configs/sgg_res101_joint.yaml --algorithm $ALGORITHM
```

Multi-GPU training:
```
python -m torch.distributed.launch --nproc_per_node=$NGPUS main.py --config-file configs/sgg_res101_joint.yaml --algorithm $ALGORITHM
```
where NGPUS is the number of gpus available. ALGORIHM is the scene graph generation model name.

## Evaluate

### Evaluate scene graph frequency baseline model:

In this case, you do not need any sgg model checkpoints. To get the evaluation result, object detector is enough. Run the following command:
```
python main.py --config-file configs/sgg_res101_{joint/step}.yaml --inference --use_freq_prior
```

In the yaml file, please specify the path MODEL.WEIGHT_DET for your object detector.

### Evaluate scene graph generation model:

* Scene graph generation model with resnet-101 as backbone:
```
python main.py --config-file configs/sgg_res101_{joint/step}.yaml --inference --resume $CHECKPOINT --algorithm $ALGORITHM
```

* Scene graph generation model with resnet-101 as backbone and use frequency prior:
```
python main.py --config-file configs/sgg_res101_{joint/step}.yaml --inference --resume $CHECKPOINT --algorithm $ALGORITHM --use_freq_prior
```

Similarly you can also append the ''--inference $YOUR_NUMBER'' to perform partially evaluate.

:warning: If you want to evaluate the model at your own path, just need to change the MODEL.WEIGHT_SGG to your own path in sgg_res101_{joint/step}.yaml.

### Visualization

If you want to visualize some examples, you just simple append the command with:
```
--visualize
```

## Citation

Currently in the process of publishing. Citation details coming soon

## Acknowledgement

We appreciate much the nicely organized code developed by [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) and [graph-rcnn.pytorch](https://github.com/jwyang/graph-rcnn.pytorch). Our codebase is built mostly based on it.
