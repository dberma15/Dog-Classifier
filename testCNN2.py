# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 16:33:36 2017

@author: bermads1
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 19:02:52 2017

@author: bermads1
"""
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd
from keras.layers import *
from keras.models import *
from keras.applications import *
from keras.applications.resnet50 import ResNet50
from keras.optimizers import *
from keras.regularizers import *
from keras.applications.inception_v3 import preprocess_input
import re
import datetime
import time
import os

print("Set directory")
parentdirectory="/home/bermads1/Classifier/Kaggle_Competition/"
print(os.getcwd())
onDanielsComputer=os.path.isdir(parentdirectory)
os.chdir(parentdirectory)


InceptionResNetV2Model="/home/bermads1/pretrained_models/InceptionResNetV2_weights_tf_dim_ordering_tf_kernels_notop_500x500.h5"
InceptionV3Model='/home/bermads1/pretrained_models/InceptionV3_weights_tf_dim_ordering_tf_kernels_notop_500x500.h5'
XceptionModel='/home/bermads1/pretrained_models/xception_weights_tf_dim_ordering_tf_kernels_notop_500x500.h5'
VGG16Model="/home/bermads1/pretrained_models/VGG16_weights_tf_dim_ordering_tf_kernels_notop_500x500.h5"
ResNet50Model="/home/bermads1/pretrained_models/ResNet50_weights_tf_dim_ordering_tf_kernels_notop_500x500.h5"
width=500

print("Reading in labels...")
labelfile='labels_training.csv'
labels=pd.read_csv(labelfile)
Y_names=pd.factorize(labels['breed'])[1]
print("Done.")


print("Reading in training images...")
n=len(labels)
breed = set(labels['breed'])
n_class = len(breed)
class_to_num = dict(zip(breed, range(n_class)))
num_to_class = dict(zip(range(n_class), breed))
#X = np.zeros((n, width, width, 3), dtype=np.uint8)
y = np.zeros((n, 120), dtype=np.uint8)

for i in tqdm(range(n)):
#    X[i] = cv2.resize(cv2.imread('train/%s.jpg' % labels['id'][i]), (width, width))
    y[i][class_to_num[labels['breed'][i]]] = 1
print("Done reading images.")
X=0
def get_features(MODEL, data):
    cnn_model = load_model(MODEL)
    
    inputs = Input((width, width, 3))
    x = inputs
    x = Lambda(preprocess_input, name='preprocessing')(x)
    x = cnn_model(x)
    x = GlobalAveragePooling2D()(x)
    cnn_model = Model(inputs, x)
    
    features = cnn_model.predict(data, batch_size=32, verbose=1)
    return features


print("Saving one hot encoding of Labels")
#else:
#	y=np.load("YtrainingfeaturesVGG_500x500.npz.npy")


if os.path.isfile("InceptionResNetV2trainingfeatures_500x500.npz.npy"):
    print('Loading InceptionResNetV2 features')
    InceptionResNetV2_features=np.load("InceptionResNetV2trainingfeatures_500x500.npz.npy")
else:
    print('Converting image to features using InceptionResNetV2...')
    InceptionResNetV2_features = get_features(InceptionResNetV2Model, X)
    np.save("InceptionResNetV2trainingfeatures_500x500.npz",InceptionResNetV2_features)
    print('done.')

if os.path.isfile("xceptiontrainingfeatures_500x500.npz.npy"):
   print('Loading Xception features')
   xception_features=np.load("xceptiontrainingfeatures_500x500.npz.npy")
else:
    print('Converting image to features using Xception...')
    xception_features = get_features(XceptionModel, X)
    np.save("xceptiontrainingfeatures_500x500.npz",xception_features)
    print('done.')
	
if os.path.isfile("VGGtrainingfeatures_500x500.npz.npy"):
    print('Loading VGG16 features')
    VGG_features=np.load("VGGtrainingfeatures_500x500.npz.npy")
else:
    print('Converting image to features using VGG16...')
    VGG_features = get_features(VGG16Model, X)
    np.save("VGGtrainingfeatures_500x500.npz",VGG_features)
    print('done.')
    
if os.path.isfile("ResNet50trainingfeatures_500x500.npz.npy"):
    ResNet50_features=np.load("ResNet50trainingfeatures_500x500.npz.npy")
else:
    print('Converting image to features using ResNet50...')
    ResNet50_features= get_features(ResNet50Model, X)
    np.save("ResNet50trainingfeatures_500x500.npz",ResNet50_features )
    print('done.')

if os.path.isfile("inceptionv3trainingfeatures_500x500.npz.npy"):
    print('Loading InceptionV3 features')
    InceptionV3_features =np.load("inceptionv3trainingfeatures_500x500.npz.npy")
else:
    print('Converting image to features using InceptionV3...')
    InceptionV3_features = get_features(InceptionV3Model,X)
    np.save("inceptionv3trainingfeatures_500x500.npz",InceptionV3_features)
    print('done.')



ResNet50min=ResNet50_features.min()
ResNet50max=ResNet50_features.max()

VGG16min=VGG_features.min()
VGG16max=VGG_features.max()

InceptionResNetV2min=InceptionResNetV2_features.min()
InceptionResNetV2max=InceptionResNetV2_features.max()

xceptionmin=xception_features.min()
xceptionmax=xception_features.max()

InceptionV3min=InceptionV3_features.min()
InceptionV3max=InceptionV3_features.max()


# ResNet50_features=(ResNet50_features-ResNet50min)/(ResNet50max-ResNet50min)
# VGG_features=(VGG_features-VGG16min)/(VGG16max-VGG16min)
# InceptionResNetV2_features=(InceptionResNetV2_features-InceptionResNetV2min)/(InceptionResNetV2max-InceptionResNetV2min)
# xception_features=(xception_features-xceptionmin)/(xceptionmax-xceptionmin)
# InceptionV3_features=(InceptionV3_features-InceptionV3min)/(InceptionV3max-InceptionV3min)

print("begin training")
#features = np.concatenate([ResNet50_features,InceptionResNetV2_features,xception_features,InceptionV3_features], axis=-1)
features = np.concatenate([ResNet50_features,VGG_features,InceptionResNetV2_features,xception_features,InceptionV3_features], axis=-1)
x_train=features[0:8500]
y_train=y[0:8500]
x_test=features[8500:]
y_test=y[8500:]

epochs=100
lrate=0.01
decay=lrate/(1/6*epochs)
sgd = SGD(lr=lrate, momentum=.8, decay=decay, nesterov=False)
inputs = Input(features.shape[1:])
x = inputs
x = Dropout(0.15)(x)
x = BatchNormalization()(x)
x = Dropout(0.9)(x)
x = BatchNormalization()(x)
x = Dense(n_class, activation='softmax')(x)
model = Model(inputs, x)
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
h = model.fit(x_train, y_train, batch_size=128, nb_epoch=epochs+100, validation_data=(x_test,y_test))

modelsavename="dogclassifier_Xception_InceptionV3_vgg16_resnet_inceptionresnet_500x500 "
ts=time.time()
st=datetime.datetime.fromtimestamp(ts).strftime("%Y%m%d %H%M%S")
filename=modelsavename+st+'.h5'

np.save("YtrainingfeaturesVGG_500x500 "+ st +'.npz',y)


print(filename)
model.save(filename)
X=0
x_train=0
y_train=0
x_test=0
y_test=0
features = 0
ResNet50_features=0
VGG_features=0
InceptionResNetV2_features=0
xception_features=0
InceptionV3_features=0

# #got .2976 without rescaling data (dogclassifier_Xception_InceptionV3_vgg16_resnet_inceptionresnet_500x500 20171211 182139 loss=0.2976.h5)
# epochs=100
# lrate=0.01
# decay=lrate/(1/6*epochs)
# sgd = SGD(lr=lrate, momentum=.75, decay=decay, nesterov=False)
# inputs = Input(features.shape[1:])
# x = inputs
# x = Dropout(0.15)(x)
# x = BatchNormalization()(x)
# x = Dropout(0.9)(x)
# x = BatchNormalization()(x)
# x = Dense(n_class, activation='softmax')(x)
# model = Model(inputs, x)
# model.compile(optimizer=sgd,
              # loss='categorical_crossentropy',
              # metrics=['accuracy'])
# h = model.fit(x_train, y_train, batch_size=200, nb_epoch=epochs+100, validation_data=(x_test,y_test))



files=os.listdir("/home/bermads1/Classifier/Kaggle_Competition/test")

#X_test = np.zeros((len(files), width, width, 3), dtype=np.uint8)
X_test=0
for i in tqdm(range(len(files))):
    X_test[i] = cv2.resize(cv2.imread('test/%s' % files[i]), (width, width))

if os.path.isfile("xceptiontestfeatures_500x500.npz.npy"):
    print('Loading Xception test features')
    xception_features_test=np.load("xceptiontestfeatures_500x500.npz.npy")
else:
    print('Converting images to features using Xception...')
    xception_features_test= get_features(XceptionModel, X_test)
    np.save("xceptiontestfeatures_500x500.npz",xception_features_test)
    print('done.')
    
if os.path.isfile("inceptionv3testfeatures_500x500.npz.npy"):
    print('Loading InceptionV3 test features')
    InceptionV3_features_test =np.load("inceptionv3testfeatures_500x500.npz.npy")
else:
    print('Converting images to features using InceptionV3...')
    InceptionV3_features_test = get_features(InceptionV3Model,X_test)
    np.save("inceptionv3testfeatures_500x500.npz",InceptionV3_features_test)
    print('done.')

if os.path.isfile("VGGtestfeatures_500x500.npz.npy"):
    print('Loading VGG16 test features')
    VGG_features_test=np.load("VGGtestfeatures_500x500.npz.npy")
else:
    print('Converting images to features using VGG16...')
    VGG_features_test= get_features(VGG16Model, X_test)
    np.save("VGGtestfeatures_500x500.npz",VGG_features_test)
    print('done.')

if os.path.isfile("InceptionResNetV2testfeatures_500x500.npz.npy"):
    print('Loading InceptionResNetV2 test features')
    InceptionResNetV2_features_test=np.load("InceptionResNetV2testfeatures_500x500.npz.npy")
else:
    print('Converting images to features using InceptionResNetV2...')
    InceptionResNetV2_features_test= get_features(InceptionResNetV2Model, X_test)
    np.save("InceptionResNetV2testfeatures_500x500.npz",InceptionResNetV2_features_test)
    print('done.')
if os.path.isfile("ResNet50testfeatures_500x500.npz.npy"):
    print('Loading ResNet50 test features')
    ResNet50_features_test=np.load("ResNet50testfeatures_500x500.npz.npy")
else:
    print('Converting images to features using ResNet50...')
    ResNet50_features_test= get_features(ResNet50Model, X_test)
    np.save("ResNet50testfeatures_500x500.npz",ResNet50_features_test)
    print('done.')

	
# ResNet50_features_test=(ResNet50_features_test-ResNet50min)/(ResNet50max-ResNet50min)
# VGG_features_test=(VGG_features_test-VGG16min)/(VGG16max-VGG16min)
# InceptionResNetV2_features_test=(InceptionResNetV2_features_test-InceptionResNetV2min)/(InceptionResNetV2max-InceptionResNetV2min)
# xception_features_test=(xception_features_test-xceptionmin)/(xceptionmax-xceptionmin)
# InceptionV3_features_test=(InceptionV3_features_test-InceptionV3min)/(InceptionV3max-InceptionV3min)
X_test=0

#features_test = np.concatenate([ResNet50_features_test, InceptionResNetV2_features_test, xception_features_test,InceptionV3_features_test], axis=-1)
features_test = np.concatenate([ResNet50_features_test,VGG_features_test, InceptionResNetV2_features_test, xception_features_test,InceptionV3_features_test], axis=-1)
predictions=model.predict(features_test)


files=[re.sub('.jpg','',x) for x in files]
files=[re.sub('test\\\\','',x) for x in files]
ids=pd.DataFrame({'id':files})



predictions=pd.DataFrame(predictions,columns=num_to_class.values())
predictions=pd.concat([ids,predictions],axis=1)
predictions.to_csv("Dog Class Predictions"+st+".csv",index=False)
