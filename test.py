'''
Created on Raj. 16, 1440 AH
23 march 2019
@author: norah
'''

from keras.preprocessing.image import ImageDataGenerator
#from keras.applications.vgg16 import VGG16
#from keras.applications.vgg16 import preprocess_input
import numpy as np
import pandas as pnd
import os
import scipy.io as sio
from shutil import copy
from keras.models import model_from_json
#from keras.layers import Dense
from keras import optimizers
from itertools import izip


#load pre-saved network model form json file
print("LOADING NETWORK ... ... ...")
net_file= open("train_net.json", "r")
json_net= net_file.read()
net_file.close()
net= model_from_json(json_net)

#load pre-saved training weights
print("LOADING WEIGHTS ... ... ...")
net.load_weights("train_w.h5")

sgd=optimizers.SGD(lr=0.0001, decay=0.0005, momentum=0.9)
net.compile(optimizer=sgd, loss='categorical_crossentropy')
net.summary()


#read test data using keras data generator
print("GENERATING TEST DATA ... ... ...") 
test_gen=ImageDataGenerator(rescale=1./255, brightness_range=(0.2,0.9), horizontal_flip=True) # augmentation to resemble training data
train_gen=ImageDataGenerator() # to get class labels
#while testing, batch_size must divide the test set
train_flow= train_gen.flow_from_directory(directory="./class/train/", target_size=(224,224), color_mode="rgb", batch_size=1, class_mode="categorical", shuffle=True, seed=24)
test_flow= test_gen.flow_from_directory(directory="./class/test", target_size=(224,224), color_mode="rgb", batch_size=1, class_mode=None, shuffle=False, seed=24)
stp= test_flow.n//test_flow.batch_size #step size. n is total number of test samples
print("Data set size: " + str(test_flow.n))

test_flow.reset() #generator needs to be reset before every test call
pred= net.predict_generator(test_flow, verbose=1, steps=stp)
#print pred
pred_label= np.argmax(pred, axis=1) #predicted classs label numbers

print("predicted classes label numbers: %s " % (pred_label))

#map pred_label to class name
ind= (train_flow.class_indices)
print("class indices: %s " % (ind))
lbls= dict((v,k) for k,v in ind.items())
print("class indices tuple: %s " % (lbls))
predict= [lbls[i] for i in pred_label] 
print("predicted classes: %s " % (predict))

fl= test_flow.filenames
print(fl)
crct=0 #counting correct predictions 
print("number of test files= "+str(len(fl))+", number of predictions= "+str(len(predict)))
for f,p_lbl in izip(fl,predict):
    f_lbl=f.split('_',1)[0]
    f_lbl=f_lbl.split('/',1)[1]
    #print("f_lbl: "+f_lbl+", p_lbl: "+p_lbl)
    #print(f_lbl==p_lbl)
    if f_lbl==p_lbl:
        crct=crct+1
print("correct predictions "+ str(crct) + "out of "+str(len(predict)))
test_acc= crct/(len(predict)*1.0)*100
print("test accuracy= "+str(test_acc))

#saving results 
print("SAVING RESULTS ... ... ...")
fl= test_flow.filenames
res= pnd.DataFrame({"Filename":fl, "Predictions": predict})
res.to_csv("test_results.csv", index=False)
print("TEST DONE!")

