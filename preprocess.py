'''
Created on Jum. II 30, 1440 AH

@author: norah
'''
import os
import cv2
import numpy as np
import random
import scipy
import scipy.ndimage
import scipy.spatial
import scipy.io as sio
from shutil import copy, move



##### adjusting img size with padding #######
#imgs are of differens sizes, each will be segmented into 224X224, 
#number of segments depend on original img size
#if one side of an img is less than 224 it will be padded to reach 224
def img_pad(img):
    h, w, c =img.shape #read img height, width, number of channels
    if w<seg_w:
        wp=(seg_w-h) #padding  
        img=cv2.copyMakeBorder(img,0,0,0,wp,cv2.BORDER_CONSTANT,value=0)
        w=seg_w
    else:
        rem_w=w%seg_w #width remainder to pad the last segment
        if rem_w != 0:
            img=cv2.copyMakeBorder(img,0,0,0,(seg_w-rem_w),cv2.BORDER_CONSTANT,value=0)
            w=w+(seg_w-rem_w)
    num_seg_w=w/seg_w #number of segments horizontally
    
    if h<seg_h:
        hp=(seg_h-h) #padding  
        img=cv2.copyMakeBorder(img,0,hp,0,0,cv2.BORDER_CONSTANT,value=0)
        h=seg_h
    else:
        rem_h=h%seg_h #high remainder to pad the last segment
        if rem_h != 0:
            img=cv2.copyMakeBorder(img,0,(seg_h-rem_h),0,0,cv2.BORDER_CONSTANT,value=0)
            h=h+(seg_h-rem_h)
    num_seg_h=h/seg_h #number of segments vertically
    
    return [img,num_seg_h,num_seg_w]



def segment(img_dir,gt_dir,seg_img_path,seg_gt_path):
    img_list=os.listdir(img_dir)
    for f in img_list:
        #img_name=img_list[f]
        if f==".DS_Store":
            continue
        img_name=f
        img_path=os.path.join(img_dir,img_name)
        img=cv2.imread(img_path) #read as rgb

        gt_name='GT_'+img_name.split('.')[0]+'.mat'
        gt=sio.loadmat(os.path.join(gt_dir,gt_name))['image_info'][0][0][0][0][0] #access location matrix from dictionary
         
         
        ##### adjusting img size with padding #######
        f_img,num_seg_h,num_seg_w=img_pad(img)        
         
        ##### create gt dot-map (zeroz and ones) #####
        f_h,f_w, c =f_img.shape #full high and width of image after padding
        dot_map=np.zeros((f_h,f_w))
        len=gt.shape[0]
        #print('f_h: '+str(f_h)+' ,f_w: '+str(f_w)+' ,len '+str(len))
        for i in range(0,len):
            x=int(gt[i,0])
            y=int(gt[i,1])
            dot_map[y,x]=1 #mark 1 for every head location
          
          
        ##### segment images and dot_map 
        ##### & create density map for each dot_map segment 
        ##### & save segments 
        count=1 #counter for naming segments
        for i in range(0,f_h,seg_h):
            for j in range(0,f_w,seg_w):
                seg_img=f_img[i:i+seg_h,j:j+seg_w] #create image segment
                seg_dot_map=dot_map[i:i+seg_h,j:j+seg_w] #create dot_map segment
                #create density map of seg_dot_map
                  
                #seg_den_map= create_density_map(seg_dot_map)
                  
                # saving segments
                img_seg_name=img_name.split('.')[0]+'_'+str(count)+'.png'
                gt_seg_name=gt_name.split('.')[0]+'_'+str(count)+'.mat'
                img_done=cv2.imwrite(os.path.join(seg_img_path,img_seg_name),seg_img) #saved as grayscale
                #if img_done: print("done img "+img_seg_name)
                #saving 1, 0 seg_dot_map for trans_learn
                sio.savemat(os.path.join(seg_gt_path,gt_seg_name),{'dot_map':seg_dot_map})
                #gt_done=sio.savemat(os.path.join(seg_gt_path,gt_seg_name),{'den_map':seg_den_map})
                #if gt_done: print("done gt "+gt_seg_name)
                count=count+1


          
def classify_data(src_img_path,src_gt_path,emp_dir,low_dir,mod_dir,high_dir,low,mod,high=None):
#preparation for transfer learning of vgg
#to separate image segments into 3 classes (folders) based on dot_map count of each segment    
    print("... ... ... copying data into corresponding classes")
    
    gt_files = [f for f in os.listdir(src_gt_path) if os.path.isfile(os.path.join(src_gt_path, f))]
    
    #count ranges for classes set with function call
    #class_emp=0 #image does font include any human being
    class_low=low
    class_mod=mod
    class_high=high #if None; any range other than (higher) previously defined ranges
    
    for i, f in enumerate(gt_files):
        if f==".DS_Store":
            continue
        gt=sio.loadmat(os.path.join(src_gt_path, f))['dot_map']
        count=np.count_nonzero(gt)
        
        img_name=f.split('_',1)[1].replace('.mat','.png')
        if count==0:
            copy(os.path.join(src_img_path,img_name),emp_dir)
        else:
            if count in class_low: 
                copy(os.path.join(src_img_path,img_name),low_dir)
            else: 
                if count in class_mod: 
                    copy(os.path.join(src_img_path,img_name),mod_dir)
                else: 
                    copy(os.path.join(src_img_path,img_name),high_dir)
    print("... ... ... done.")

def rename_test_data(src,dst,dmap,low,mod,high=None):
    print("... ... ... renaming files based on count ... ... ...")
    
    gt_files = [f for f in os.listdir(dmap) if os.path.isfile(os.path.join(dmap, f))]
    
    #count ranges for classes set with function call
    #class_emp=0 #image does font include any human being
    class_low=low
    class_mod=mod
    class_high=high #if None; any range higher than previously defined ranges
    
    for i, f in enumerate(gt_files):
        if f==".DS_Store":
            continue
        gt=sio.loadmat(os.path.join(dmap, f))['dot_map']
        count=np.count_nonzero(gt)
        
        img_name=f.split('_',1)[1].replace('.mat','.png')
        if count==0:
            class_name="emp_" + img_name
            os.rename(os.path.join(src,img_name),os.path.join(dst,class_name))
        else:
            if count in class_low: 
                class_name="low_" + img_name
                os.rename(os.path.join(src,img_name),os.path.join(dst,class_name))
            else: 
                class_name="mod_" + img_name
                if count in class_mod: 
                    os.rename(os.path.join(src,img_name),os.path.join(dst,class_name))
                else: 
                    class_name="high_" + img_name
                    os.rename(os.path.join(src,img_name),os.path.join(dst,class_name))
    print("... ... ... done, "+str(i)+" files.")

if __name__ == '__main__':

    seg_h= 224 #width and hight of segment based on network input spec.
    seg_w= 224
    
    #origin data path
    train_img_dir="./ShanghaiTech/part_A/train_data/images"
    train_gt_dir="./ShanghaiTech/part_A/train_data/ground-truth"
    test_img_dir="./ShanghaiTech/part_A/test_data/images"
    test_gt_dir="./ShanghaiTech/part_A/test_data/ground-truth"
    
    
    #destination directory where training dataset segments will be stored
    train_seg_img_path="./seg/train/img"
    train_dot_map_path='./seg/train/dmap'
    #train_seg_gt_path="./seg/train/denmap"
    if not os.path.exists(train_seg_img_path): os.makedirs(train_seg_img_path)
    if not os.path.exists(train_dot_map_path): os.makedirs(train_dot_map_path)
    #if not os.path.exists(train_seg_gt_path): os.makedirs(train_seg_gt_path)
    
    #destination directory where testing dataset segments will be stored
    test_seg_img_path="./seg/test/img"
    test_dot_map_path='./seg/test/dmap'
    #test_seg_gt_path="./seg/test/denmap"
    if not os.path.exists(test_seg_img_path): os.makedirs(test_seg_img_path)
    if not os.path.exists(test_dot_map_path): os.makedirs(test_dot_map_path)
    #if not os.path.exists(test_seg_gt_path): os.makedirs(test_seg_gt_path)
    
    #train classification directories
    train_class_emp_dir='./class/train/emp'
    train_class_low_dir='./class/train/low'
    train_class_mod_dir='./class/train/mod'
    train_class_high_dir='./class/train/high'
    if not os.path.exists(train_class_emp_dir): os.makedirs(train_class_emp_dir)
    if not os.path.exists(train_class_low_dir): os.makedirs(train_class_low_dir)
    if not os.path.exists(train_class_mod_dir): os.makedirs(train_class_mod_dir)
    if not os.path.exists(train_class_high_dir): os.makedirs(train_class_high_dir)
    
    test_class_dir='./class/test/img'
    if not os.path.exists(test_class_dir): os.makedirs(test_class_dir)
    
    #preparing training dataset
    print("SEGMENT TRAIN DATA ... ... ...")
    segment(train_img_dir,train_gt_dir,train_seg_img_path,train_dot_map_path)
    #preparing testing dataset
    print("SEGMENT TEST DATA ... ... ...")
    segment(test_img_dir,test_gt_dir,test_seg_img_path,test_dot_map_path)
    
    print("CLASSIFICATION PROCESS ... ... ... ")
    #train classification
    
    rlow=range(1,50)
    rmod=range(51,200)
    
    print("classifying train data ... ... ...")
    classify_data(train_seg_img_path,train_dot_map_path,train_class_emp_dir,train_class_low_dir,train_class_mod_dir,train_class_high_dir,rlow,rmod)
    
    #prepare test data
    print("copying test data ... ... ...")
    dmap="./seg/test/dmap"
    rename_test_data(test_seg_img_path,test_class_dir,test_dot_map_path,rlow,rmod)
    
        
    print("PRE-PROCESSING DONE.")
    
