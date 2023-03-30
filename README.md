# Final Project: Garbage Detection
This repository contains the code of the final project for the Computer Vision: 3D Reconstruction class at Heidelberg University. The group members are Josch Hagedorn, Ibrahima Sory Kourouma, and Christian Teutsch. The contributions to the written report are as follows: Josch (Theoretical background CNNs and Object Detection, State of the art), Ibrahima (Abstract, Introduction, Methodology, Results, Discussion, Conclusion), Christian (Theoretical backround Detectors, R-CNN, Fast R-CNN, Faster R-CNN, YOLO, Detector Comparison). The contributions to the software are as follows: Josch (Implementation using Google Colab, working with the Webcam), Ibrahima (Implementation Pytorch, Performance Tests, Demo, Github), Christian (Implementation TensorFlow, Researching different Models).

## Introduction
Our mini-research project aims to develop a garbage classification system for efficient recycling. The system utilizes a webcam feed to detect and highlight a garbage item and then classify it into the appropriate bin (e.g. plastic, paper, glass, etc.). One of the difficulties of this project lies in the fact that it is not based on individual pictures but, on the live feed of the camera. For that purpose, the item of garbage is tagged in the live video to show the user that the correct item was classified and not some other item in the background.

## Setting up project locally

It is important to note that the environment in which the project is executed must not contain the google.colab library, as it serves as an indicator of whether the code is executed locally or in the browser. This has an impact on the demo project. See in the class application (garbage_detector/application__ini__.py)

### Unix

You can set up a python virtual environment.
See [Venv Tutorial](https://mothergeo-py.readthedocs.io/en/latest/development/how-to/venv.html).

In the virtual environment

``
pip install -r requirements.txt
``

is executed to install all the necessary libraries that are needed. 

## Windows 

Follow [PyTorch installation guide](https://pytorch.org/get-started/locally/) for installation in Windows.

After installation of PyTorch, simply execute 

``
pip install -r requirements.txt
``

## Structure of the demo.ipynp

The demo.ipynb can be divided into 5 main parts, whereas part 1 and 2 concern the imports of necessary libraries and the setup of the computing device (cuda or CPU).

The parameter num_workers is set delibaretly to 0, as problems can occur under Windows if num_workers > 0.
See [PyTorch issue](https://discuss.pytorch.org/t/errors-when-using-num-workers-0-in-dataloader/97564)

It is possible to run this project under WSL2 and increase the number of num_workers. But one should be aware that accessing the webcam under WSL2, which is necessary for the demo, can be difficult and in many cases will not work even with workarounds found on the internet.

See [WSL Webcam issue](https://github.com/microsoft/WSL/issues/6211).

So, one could run the training under WSL2 and the demo under Windows.

The other 3 main parts can be executed independently of each other.

### CNN Garbage Classification Benchmark

The *Setup* block needs always to be executed.
The results of the training are already part of this project (results/classification/benchmark.csv), so the
*Show test results* block can be executed without starting the *Run Benchmark* block.

### Faster RCCN Garbage Detection Benchmark

The *Setup* block needs to be executed first, as well.
The results are under results/detection/benchmark.csv
and can therefore be displayed in the *Show test results* block without running the benchmark again.

### Garbage Detection Demo

The *Setup* block needs to be executed first, as well.
The *Train* block is optional and is not important to execute.
When the *Run Application* block is executed, the model that is used for the demo, downloads its weights from Google Drive. The parameter *force_download* is set to True, which means that the weights are always downloaded even if they can be found locally. So once the download has taken place, the parameter can be set to False.

## Reproducibility
The simplest way to reproduce the results is by using the Google Colab notebook. Make sure to use the GPU runtime to have the best performance experience.

See [Garbage Classifier Google Colab](https://colab.research.google.com/drive/1B7BdAqk0vazvmtAoMTy6LvGNu4T1dwkb?usp=sharing)


While working on our project, we also worked on a proof of concept using TensorFlow. We sometimes experienced better performance here. In the notebook itself an older model can be chosen. This model was not trained as long as the final TensorFlow model and can perform better. We suspect that the this model better generalises due to less training and less overfitting. See
[Garbage Classifier with TensorFlow Google Colab](https://colab.research.google.com/drive/1oEVayMMwGKNntMHY51VV-3IrWvUNuE1y?usp=sharing)

Just go to the Google Colab Menu "Runtime" then click on "Run all" this will take approximately 5 minutes for the TensorFlow PoC.
