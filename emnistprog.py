import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
# import zipfile
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
#import sklearn
import imutils
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.datasets import mnist
# MNIST
(xt,yt),(xte,yte)=mnist.load_data()
digitsdata=np.vstack([xt,xte])
digitslabels=np.hstack([yt,yte])
#Alphabets
az=pd.read_csv('A_Z Handwritten Data.csv').astype('float32')
alphabetdata=az.drop('0',axis=1)
alphabetlabels=az['0']
alphabetdata=np.reshape(alphabetdata.values,(alphabetdata.shape[0],28,28))
alphabetlabels+=10
idata=np.vstack([alphabetdata,digitsdata])
ilabels=np.hstack([alphabetlabels,digitslabels])
# Alphabets2
az2=pd.read_csv('azdataset2.csv')
azdata=az2.drop('0',axis=1)
azdata=np.reshape(azdata.values,(azdata.shape[0],28,28))
azlabel=az2['0']
azlabel+=10
idata=np.vstack([idata,azdata])
ilabels=np.hstack([ilabels,azlabel])
# # EMNIST Balanced
df=pd.read_csv("emnist-balanced-train.csv")
data=df.drop("45",axis=1)
labels=df["45"]
data=np.reshape(data.values,(data.shape[0],28,28))
test=pd.read_csv("emnist-balanced-test.csv")
testdata=test.drop("41",axis=1)
testlabel=test["41"]
testdata=np.reshape(testdata.values,(testdata.shape[0],28,28))
#Custom Dataset
smallalabels=[]
smallblabels=[]
smalldlabels=[]
smallelabels=[]
smallflabels=[]
smallglabels=[]
smallhlabels=[]
smallnlabels=[]
smallqlabels=[]
smallrlabels=[]
smalltlabels=[]
smalla=pd.read_csv('newa.csv').astype('float32')
for i in range(len(smalla)):
    smallalabels.append(0)
smalla=np.reshape(smalla.values,(smalla.shape[0],28,28))
smallb=pd.read_csv('newb.csv').astype('float32')
for i in range(len(smallb)):
    smallblabels.append(1)
smallb=np.reshape(smallb.values,(smallb.shape[0],28,28))
smalld=pd.read_csv('newd.csv').astype('float32')
for i in range(len(smalld)):
    smalldlabels.append(2)
smalld=np.reshape(smalld.values,(smalld.shape[0],28,28))
smalle=pd.read_csv('newe.csv').astype('float32')
for i in range(len(smalle)):
    smallelabels.append(3)
smalle=np.reshape(smalle.values,(smalle.shape[0],28,28))
smallf=pd.read_csv('newf.csv').astype('float32')
for i in range(len(smallf)):
    smallflabels.append(4)
smallf=np.reshape(smallf.values,(smallf.shape[0],28,28))
smallg=pd.read_csv('newg.csv').astype('float32')
for i in range(len(smallg)):
    smallglabels.append(5)
smallg=np.reshape(smallg.values,(smallg.shape[0],28,28))
smallh=pd.read_csv('newh.csv').astype('float32')
for i in range(len(smallh)):
    smallhlabels.append(6)
smallh=np.reshape(smallh.values,(smallh.shape[0],28,28))
smalln=pd.read_csv('newn.csv').astype('float32')
for i in range(len(smalln)):
    smallnlabels.append(7)
smalln=np.reshape(smalln.values,(smalln.shape[0],28,28))
smallq=pd.read_csv('newq.csv').astype('float32')
for i in range(len(smallq)):
    smallqlabels.append(8)
smallq=np.reshape(smallq.values,(smallq.shape[0],28,28))
smallr=pd.read_csv('newa.csv').astype('float32')
for i in range(len(smallr)):
    smallrlabels.append(9)
smallr=np.reshape(smallr.values,(smallr.shape[0],28,28))
smallt=pd.read_csv('newa.csv').astype('float32')
for i in range(len(smallt)):
    smalltlabels.append(10)
smallt=np.reshape(smallt.values,(smallt.shape[0],28,28))
custdata=np.vstack([smalla,smallb,smalld,smalle,smallf,smallg,smallh,smalln,smallq,smallr,smallt])
custlabels=np.hstack([smallalabels,smallblabels,smalldlabels,smallelabels,smallflabels,smallglabels,smallhlabels,smallnlabels,smallqlabels,smallrlabels,smalltlabels])
custlabels+=36
# EMNIST Byclass
# byclasstrain=pd.read_csv("emnist-byclass-train.csv")
# bytraindata=byclasstrain.drop("35",axis=1)
# bytrainlabels=byclasstrain["35"]
# bytraindata=np.reshape(bytraindata.values,(bytraindata.shape[0],28,28))
# byclasstest=pd.read_csv("emnist-byclass-test.csv")
# bytestdata=byclasstest.drop("18",axis=1)
# bytestlabels=byclasstest["18"]
# bytestdata=np.reshape(bytestdata.values,(bytestdata.shape[0],28,28))
# fdata=np.vstack([bytraindata,bytestdata]) # fbydata
# flabel=np.hstack([bytrainlabels,bytestlabels]) #fbylabel
fbdata=np.vstack([data,testdata])
fblabel=np.hstack([labels,testlabel])
# # fdata=np.vstack([fbydata,fbdata])
# # flabel=np.hstack([fbylabel,fblabel])
fdata=np.vstack([idata,fbdata])
flabel=np.hstack([ilabels,fblabel])
fdata=np.vstack([fdata,custdata])
flabel=np.hstack([flabel,custlabels])
fdata=np.array(fdata,dtype='float32')
fdata=np.expand_dims(fdata,axis=-1)
fdata=fdata/255.0
le=LabelBinarizer()
labels=le.fit_transform(flabel)
class_total=labels.sum(axis=0)
class_weight={}
for i in range(0,len(class_total)):
  class_weight[i]=class_total.max()/class_total[i]
xtrain,xtest,ytrain,ytest=train_test_split(fdata,labels,test_size=0.2,random_state=1,stratify=labels)
from keras.preprocessing.image import ImageDataGenerator
augmentation=ImageDataGenerator(rotation_range=20,zoom_range=0.2,width_shift_range=0.2,height_shift_range=0.2,horizontal_flip=False,shear_range=0.2)
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Flatten,Dense
from keras.callbacks import ModelCheckpoint
n1=Sequential()
n1.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=(28,28,1)))
n1.add(MaxPool2D(pool_size=(2,2)))
n1.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same'))
n1.add(MaxPool2D(pool_size=(2,2)))
n1.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu',padding='same'))
n1.add(MaxPool2D(pool_size=(2,2)))
n1.add(Flatten())
n1.add(Dense(64,activation='relu'))
n1.add(Dense(128,activation='relu'))
n1.add(Dense(47 ,activation='softmax'))
n1.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
file_model='newocr.model'
epochs=25
batch_size=128
checkpoint=ModelCheckpoint(file_model,monitor='val_loss',verbose=1,save_best_only=True) # this is used the save the best models as each epoch ends
history=n1.fit(augmentation.flow(xtrain,ytrain,batch_size=batch_size,),validation_data=(xtest,ytest),steps_per_epoch=len(xtrain)//batch_size,epochs=epochs,class_weight=class_weight,verbose=1,callbacks=[checkpoint])
