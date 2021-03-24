import tensorflow as tf
#import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D,GlobalAveragePooling2D,Dropout, Flatten,InputLayer,BatchNormalization,Lambda,Input
import numpy as np
import cv2
import heapq
from models import get_model,get_inception_model,get_xception_model
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_v3_preprocess_input
from tensorflow.keras.applications.xception import  preprocess_input as xception_preprocess_input
from Functions import get_inception_features,get_xception_features,path_to_tensor
from collections import OrderedDict
from flask import Blueprint,jsonify, request
from flask_restful import Api,Resource
import tensorflow as tf
from werkzeug.utils import secure_filename
from requests import Request, Session
import sys
import os
import json
import dlib
from imutils import face_utils
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
import io
import requests
import PIL
import numpy as np
import cv2
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy_utils import ScalarListType
from sqlalchemy.exc import IntegrityError
import pickle
import gc
from requests_toolbelt.multipart.encoder import MultipartEncoder
import time
import validators
from threading import Thread
tf.keras.backend.clear_session()


def cost(old,new):
  diff=new-old
  abs_diff=diff**2
  n=len(abs_diff)*2
  diff_sum=sum(abs_diff)
  ans=diff_sum/n
  print(ans)
  return ans


def url_to_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    imgarr = np.array(img)
    rgb_image = cv2.cvtColor(imgarr,cv2.COLOR_BGR2RGB)
    return rgb_image

    
def image_matching(query_url,image_urls):
    print("matching..")
    valid=validators.url(query_url)
    if valid:
        print(valid)
        rgb_image=url_to_image(query_url)
        print("get rgb image")
        inception_features = get_inception_features(path_to_tensor(rgb_image))     # extract bottleneck features
        xception_features=get_xception_features(path_to_tensor(rgb_image))
        print("Concate features")
        final_features_query = np.concatenate([inception_features,xception_features], axis = 1)
    else:
        return jsonify({"api_status":400,"msg":"query Url is not valid"})
    cost_list=[]
    for image_url in image_urls:
        valid=validators.url(image_url)
        if valid:
            print(valid)
            rgb_image=url_to_image(image_url)
            inception_features = get_inception_features(path_to_tensor(rgb_image))     # extract bottleneck features
            xception_features=get_xception_features(path_to_tensor(rgb_image))
            print("Concate features")
            final_features = np.concatenate([inception_features,xception_features], axis = 1)
            print("final features")
            cost_list.append(cost(final_features_query[0],final_features[0]))
        else:
            return jsonify({"api_status":400,"msg":"Url is not valid"})

    print(image_urls,len(image_urls))
    print(cost_list,len(cost_list))
    d=dict(zip(cost_list,image_urls)) 
    d=OrderedDict(sorted(d.items()))
    sort_urls=[]
    sort_costs=[]
    for k,v in d.items():
        sort_costs.append(k)
        sort_urls.append(v)
    return sort_urls,sort_costs
    



