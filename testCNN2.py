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
parentdirectory="C:\\Users\\bermads1\\Documents\\Daniel\\Dog classifier"
onDanielsComputer=os.path.isdir(parentdirectory)
os.chdir(parentdirectory)
labelfile='labels_training.csv'
width=250

labels=pd.read_csv(labelfile)
Y_names=pd.factorize(labels['breed'])[1]

n=len(labels)
breed = set(labels['breed'])
n_class = len(breed)
class_to_num = dict(zip(breed, range(n_class)))
num_to_class = dict(zip(range(n_class), breed))
X = np.zeros((n, width, width, 3), dtype=np.uint8)
y = np.zeros((n, 120), dtype=np.uint8)

for i in tqdm(range(n)):
    X[i] = cv2.resize(cv2.imread('train/%s.jpg' % labels['id'][i]), (width, width))
    y[i][class_to_num[labels['breed'][i]]] = 1

def get_features(MODEL, data=X):
    cnn_model = MODEL(include_top=False, input_shape=(width, width, 3), weights='imagenet')
    
    inputs = Input((width, width, 3))
    x = inputs
    x = Lambda(preprocess_input, name='preprocessing')(x)
    x = cnn_model(x)
    x = GlobalAveragePooling2D()(x)
    cnn_model = Model(inputs, x)
    
    features = cnn_model.predict(data, batch_size=64, verbose=1)
    return features
    
if not os.path.isfile("YtrainingfeaturesVGG_250x250.npz.npy"):
    np.save("YtrainingfeaturesVGG_250x250.npz",y)

if os.path.isfile("xceptiontrainingfeatures_250x250.npz.npy"):
    xception_features=np.load("xceptiontrainingfeatures_250x250.npz.npy")
else:
    xception_features = get_features(Xception, X)
    np.save("xceptiontrainingfeatures_250x250.npz",xception_features)
    
if os.path.isfile("inceptionv3trainingfeatures_250x250.npz.npy"):
    InceptionV3_features =np.load("inceptionv3trainingfeatures_250x250.npz.npy")
else:
    InceptionV3_features = get_features(InceptionV3,X)
    np.save("inceptionv3trainingfeatures_250x250.npz",InceptionV3_features)

if os.path.isfile("VGGtrainingfeatures_250x250.npz.npy"):
    VGG_features=np.load("VGGtrainingfeatures_250x250.npz.npy")
else:
    VGG_features = get_features(VGG16, X)
    np.save("VGGtrainingfeatures_250x250.npz",VGG_features)


features = np.concatenate([VGG_features,xception_features,InceptionV3_features], axis=-1)
x_train=features[0:9000]
y_train=y[0:9000]
x_test=features[9000:]
y_test=y[9000:]
epochs=200
lrate=0.003
decay=lrate/epochs
sgd = SGD(lr=lrate, momentum=.9, decay=decay, nesterov=False)
adam=Adam(lr=lrate,decay=decay)
inputs = Input(features.shape[1:])
x = inputs
x = Dropout(0.85)(x)
x = Dense(n_class, activation='softmax')(x)
model = Model(inputs, x)
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
h = model.fit(x_train, y_train, batch_size=128, nb_epoch=epochs, validation_data=(x_test,y_test))

for i in range(0,2):
    h = model.fit(features, y, batch_size=2500, nb_epoch=1, validation_split=0.1)

files=os.listdir("C:\\Users\\bermads1\\Documents\\Daniel\\Dog classifier\\test")

X_test = np.zeros((len(files), width, width, 3), dtype=np.uint8)
for i in tqdm(range(len(files))):
    X_test[i] = cv2.resize(cv2.imread('test/%s' % files[i]), (width, width))

if os.path.isfile("xceptiontestfeatures_250x250.npz.npy"):
    xception_features_test=np.load("xceptiontestfeatures_250x250.npz.npy")
else:
    xception_features_test= get_features(Xception, X_test)
    np.save("xceptiontestfeatures_250x250.npz",xception_features_test)
    
if os.path.isfile("inceptionv3testfeatures_250x250.npz.npy"):
    InceptionV3_features_test =np.load("inceptionv3testfeatures_250x250.npz.npy")
else:
    InceptionV3_features_test = get_features(InceptionV3,X_test)
    np.save("inceptionv3testfeatures_250x250.npz",InceptionV3_features_test)

if os.path.isfile("VGGtestfeatures_250x250.npz.npy"):
    VGG_features_test=np.load("VGGtestfeatures_250x250.npz.npy")
else:
    VGG_features_test= get_features(VGG16, X_test)
    np.save("VGGtestfeatures_250x250.npz",VGG_features_test)



    
features_test = np.concatenate([VGG_features_test, xception_features_test,InceptionV3_features_test], axis=-1)
predictions=model.predict(features_test)
  
modelsavename="dogclassifier_Xception_InceptionV3_vgg16_250x250 loss=0.5755"

files=[re.sub('.jpg','',x) for x in files]
files=[re.sub('test\\\\','',x) for x in files]
ids=pd.DataFrame({'id':files})

ts=time.time()
st=datetime.datetime.fromtimestamp(ts).strftime("%Y%m%d %H%M%S")
filename=modelsavename+st+'.h5'
model.save(filename)
ts=time.time()
st=datetime.datetime.fromtimestamp(ts).strftime("%Y%m%d %H%M%S")


predictions=pd.DataFrame(predictions,columns=num_to_class.values())
predictions=pd.concat([ids,predictions],axis=1)
predictions.to_csv("Dog Class Predictions"+st+".csv",index=False)