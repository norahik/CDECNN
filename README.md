# CDECNN
Crowd Density Estimation CNN

updated on 23 march 2019

project built on shanghaiTech partA dataset. transfere learning vgg-16 to predict crowd density class.

files description: -preprocess:segment train and test images and dotmaps into (224x224), classify segmented train data into (empty, low, moderate, high) density classes based on segment crowd count, rename test files and copy to proper directory. -train: training modified vgg16 model on 4 classes (empty, low, moderate, high) lr=0.001, early stopping at epoch 50. -test: test data augmented to resemble train data, accuracy 77%.
