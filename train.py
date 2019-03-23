'''
Created on Raj. 16, 1440 AH
23 march 2019
@author: norah
'''

#train whole from scratch, ls=0.001, epochs=50

import os
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
import scipy.io as sio
from shutil import copy
import random
from keras.models import Model
from keras.layers import Dense, Dropout
from keras import optimizers
from tensorflow.contrib.metrics.python.metrics.classification import accuracy
from matplotlib import pyplot

        
if __name__ == '__main__':

    
    #here we redefine the structure of vgg16 to fit our 3 classes by changing the last FC layer
    print("BUILDING NETWORK ... ... ...")
    vgg_net=VGG16(weights=None, include_top=True) #include_top True will include the last 3 FC layers, False will not
    drop1=Dropout(0.5)(vgg_net.layers[-3].output)
    fc2=Dense(4096, activation='relu', name='fc2')(drop1)
    drop2=Dropout(0.5)(fc2)
    dens_prd=Dense(4, activation='softmax', name='predictions')(drop2)
    net=Model(input=vgg_net.input, output=dens_prd)
    
    #initializing training parameters
    sgd=optimizers.SGD(lr=0.001, decay=0.0005, momentum=0.9)
    #compiling the network and defining the loss method
    net.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    net.summary()
     
     
    #read pre-classified images under train_class, using keras data generator flow_from_directory()   
    print("GENERATING DATA ... ... ...")
    
    
    train_gen=ImageDataGenerator(validation_split=0.2, rescale=1./255, brightness_range=(0.2,0.9), horizontal_flip=True,featurewise_center=True, featurewise_std_normalization=True)
    
    train_flow= train_gen.flow_from_directory(directory="./class/train/", target_size=(224,224), color_mode="rgb", batch_size=1, class_mode="categorical", shuffle=True, seed=24, subset="training")
    valid_flow= train_gen.flow_from_directory(directory="./class/train/", target_size=(224,224), color_mode="rgb", batch_size=1, class_mode="categorical", shuffle=True, seed=24, subset="validation")
    stp_train= train_flow.n//train_flow.batch_size #step size. n is total number of train samples
    stp_valid= valid_flow.n//valid_flow.batch_size
    print("train set size: " + str(train_flow.n)+", valid set size: "+str(valid_flow.n))
    print("stp_valid: "+str(stp_valid))
    
    #train the network
    print("STARTING TO TRAIN ... ... ...")
    train_history= net.fit_generator(generator=train_flow, steps_per_epoch=stp_train, validation_data=valid_flow, validation_steps=stp_valid, epochs=50)
    print("DONE TRAINING.")
     
    print("STARTING EVALUATION ... ... ...")
    evalu=net.evaluate_generator(generator=valid_flow, steps=stp_valid)
     
    
    #plot history
    pyplot.plot(train_history.history['acc'], label='training')
    pyplot.plot(train_history.history['val_acc'], label='validation')
    pyplot.legend()
    pyplot.show()

    #saving history
    hf=open("train_history.txt","w")
    hf.seek(0)
    hf.write("train history: " + str(train_history.history)+ "\n accuracy: " + str(np.mean(train_history.history['acc'])))
    hf.close()
     
    #saving weights
    net.save_weights("train_w.h5")
    print("Training Weights Saved.")
     
    #saving model structure
    json_net= net.to_json()
    with open("train_net.json", "w") as jf:
        jf.seek(0)
        jf.write(json_net)
        print("Net Structure Saved.")
        
    #display loss history
    print("Training History:")
    print(train_history.history)
    print("Evaluation:")
    print("loss: ", evalu[0], ", accuracy: ", evalu[1])    