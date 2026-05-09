# Drone_Detection-Fastercnn_fpn-CustomDataset
A drone detector that uses the Roboflow dataset https://universe.roboflow.com/drone-detection-pexej/drone-detection-data-set-yolov7/dataset/1 as its training dataset and simple programming. It requires very few resources and training can be run on a personal computer.

Installation:

If they are not already installed, the following programs are required:

pip install torch

pip install torchvision

pip install requests

pip install packaging

pip install pyparsing

pip install cycler

pip install python-dateutil (NOTE: NOT dateutil)

pip install kiwisolver

pip install importlib_resources

Unzip the Test2.zip file containing the test data.

Since these are common modules in a Python development environment, you can try running the project and installing only the modules that generated the "module not found" error.

Download the project files to your hard drive. In that folder, download the COCO JSON version of the file https://universe.roboflow.com/drone-detection-pexej/drone-detection-data-set-yolov7/dataset/1 as a .zip file and extract it.

Training:

Run the program:

python train.py

The program automatically generates the fasterrcnn_fpn.pth model, which produces satisfactory results with only 5 epochs.

Verification of results:

python test.py

Comments:

To run on a personal computer with 16GB of RAM, the number of records in the training file has been reduced to 500 (this can be modified by changing line 117 of the train.py program), and the number of epochs has been reduced to 5. Despite this, the results are satisfactory, and based on the tests performed, it is the best at detecting drones among the projects listed in the references. However, it has a significant problem: it sometimes detects people, cars, and other objects from COCO, which seems to indicate that the model inherits weights from the pre-trained COCO model at some point, or that the number of training epochs was insufficient.

![Figure 1](https://github.com/ablanco1950/Drone_Detection-Fastercnn_fpn-CustomDataset/blob/main/Figure_80.png)

![Figure 1](https://github.com/ablanco1950/Drone_Detection-Fastercnn_fpn-CustomDataset/blob/main/Figure_76.png)

![Figure 1](https://github.com/ablanco1950/Drone_Detection-Fastercnn_fpn-CustomDataset/blob/main/Figure_88.png)

References:

https://universe.roboflow.com/drone-detection-pexej/drone-detection-data-set-yolov7/dataset/1


