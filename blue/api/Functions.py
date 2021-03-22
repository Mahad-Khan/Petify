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
tf.keras.backend.clear_session()

def get_inception_features(image):
    #from tensorflow.keras.applications.inception_v3 import preprocess_input
    print("Incption Loading")
    base_model =get_inception_model()
    print("Sequential")
    model = Sequential()
    print("Input Layer")
    model.add(InputLayer(input_shape = (331,331,3)))
    print("Lambda")
    model.add(Lambda(inception_v3_preprocess_input))
    print("Base model")
    model.add(base_model)
    print("Predict Feature From inception")
    feature = model.predict_on_batch(image)
    ###tf.keras.backend.clear_session()
    return feature
    
def get_xception_features(image):
    #from tensorflow.keras.applications.xception import  preprocess_input 
    print("Exception Loading")
    base_model =get_xception_model()
    print("Exception Sequential")
    model = Sequential()
    print("Exception Input Layer")
    model.add(InputLayer(input_shape = (331,331,3)))
    print("Exception Lambda")
    model.add(Lambda(xception_preprocess_input))
    print("Exception Base model")
    model.add(base_model)
    print("predict features from Exception")
    feature = model.predict_on_batch(image)
    ###tf.keras.backend.clear_session()
    return feature


def path_to_tensor(img):
    img=cv2.resize(img,(331,331))
    #img = img.astype(np.float64)
    #return np.expand_dims(img, axis=0)
    x = np.zeros([1, 331, 331, 3], dtype=np.uint8)
    x[0]=img
    return x

def predict_breed(img):
    inception_features = get_inception_features(path_to_tensor(img))     # extract bottleneck features
    xception_features=get_xception_features(path_to_tensor(img))
    print("Concate features")
    final_features = np.concatenate([inception_features,xception_features], axis = 1)
    model=get_model()
    print("Predict features from complete model")
    predicted_vector = model.predict_on_batch(final_features)       # obtain predicted vector

    tf.keras.backend.clear_session()
    return predicted_vector[0]

def return_dog_names():
    className=['affenpinscher',
                'afghan_hound',
                'airedale_terrier',
                'akita',
                'alaskan_malamute',
                'american_eskimo_dog',
                'american_foxhound',
                'american_staffordshire_terrier',
                'american_water_spaniel',
                'anatolian_shepherd_dog',
                'australian_cattle_dog',
                'australian_shepherd',
                'australian_terrier',
                'basenji',
                'basset_hound',
                'beagle',
                'bearded_collie',
                'beauceron',
                'bedlington_terrier',
                'belgian_malinois',
                'belgian_sheepdog',
                'belgian_tervuren',
                'bernese_mountain_dog',
                'bichon_frise',
                'black_and_tan_coonhound',
                'black_russian_terrier',
                'bloodhound',
                'bluetick_coonhound',
                'border_collie',
                'border_terrier',
                'borzoi',
                'boston_terrier',
                'bouvier_des_flandres',
                'boxer',
                'boykin_spaniel',
                'briard',
                'brittany',
                'brussels_griffon',
                'bull_terrier',
                'bulldog',
                'bullmastiff',
                'cairn_terrier',
                'canaan_dog',
                'cane_corso',
                'cardigan_welsh_corgi',
                'cavalier_king_charles_spaniel',
                'chesapeake_bay_retriever',
                'chihuahua',
                'chinese_crested',
                'chinese_shar-pei',
                'chow_chow',
                'clumber_spaniel',
                'cocker_spaniel',
                'collie',
                'curly-coated_retriever',
                'dachshund',
                'dalmatian',
                'dandie_dinmont_terrier',
                'doberman_pinscher',
                'dogue_de_bordeaux',
                'english_cocker_spaniel',
                'english_setter',
                'english_springer_spaniel',
                'english_toy_spaniel',
                'entlebucher_mountain_dog',
                'field_spaniel',
                'finnish_spitz',
                'flat-coated_retriever',
                'french_bulldog',
                'german_pinscher',
                'german_shepherd_dog',
                'german_shorthaired_pointer',
                'german_wirehaired_pointer',
                'giant_schnauzer',
                'glen_of_imaal_terrier',
                'golden_retriever',
                'gordon_setter',
                'great_dane',
                'great_pyrenees',
                'greater_swiss_mountain_dog',
                'greyhound',
                'havanese',
                'ibizan_hound',
                'icelandic_sheepdog',
                'irish_red_and_white_setter',
                'irish_setter',
                'irish_terrier',
                'irish_water_spaniel',
                'irish_wolfhound',
                'italian_greyhound',
                'japanese_chin',
                'keeshond',
                'kerry_blue_terrier',
                'komondor',
                'kuvasz',
                'labrador_retriever',
                'lakeland_terrier',
                'leonberger',
                'lhasa_apso',
                'lowchen',
                'maltese',
                'manchester_terrier',
                'mastiff',
                'miniature_schnauzer',
                'neapolitan_mastiff',
                'newfoundland',
                'norfolk_terrier',
                'norwegian_buhund',
                'norwegian_elkhound',
                'norwegian_lundehund',
                'norwich_terrier',
                'nova_scotia_duck_tolling_retriever',
                'old_english_sheepdog',
                'otterhound',
                'papillon',
                'parson_russell_terrier',
                'pekingese',
                'pembroke_welsh_corgi',
                'petit_basset_griffon_vendeen',
                'pharaoh_hound',
                'plott',
                'pointer',
                'pomeranian',
                'poodle',
                'portuguese_water_dog',
                'saint_bernard',
                'silky_terrier',
                'smooth_fox_terrier',
                'tibetan_mastiff',
                'welsh_springer_spaniel',
                'wirehaired_pointing_griffon',
                'xoloitzcuintli',
                'yorkshire_terrier',
                'shetland_sheepdog',
                'english_foxhound',
                'african_hunting_dog',
                'dhole',
                'dingo',
                'mexican_hairless',
                'standard_poodle',
                'miniature_poodle',
                'toy_poodle',
                'brabancon_griffon',
                'samoyed',
                'pug',
                'malamute',
                'eskimo_dog',
                'entleBucher',
                'appenzeller',
                'miniature_pinscher',
                'rottweiler',
                'kelpie',
                'malinois',
                'groenendael',
                'schipperke',
                'siberian_husky',
                'sussex_spaniel',
                'vizsla',
                'west_Highland_white_terrier',
                'scotch_terrier',
                'sealyham_terrier',
                'irish_terrier',
                'shih-Tzu',
                'japanese_spaniel',
                'redbone',
                'walker_hound',
                'wire-haired_fox_terrier',
                'whippet',
                'weimaraner',
                'soft-coated_wheaten_terrier',
                'staffordshire_bullterrier',
                'scottish_deerhound',
                'saluki',
                'blenheim_spaniel',
                'toy_terrier',
                'rhodesian_ridgeback',
                'standard_schnauzer',
                'tibetan_terrier',
                'miniature poodle',
                'harrier',
                'jack russel terrier',
                'polish lowland sheepdog',
                'dogo argentino',
                'miniature bull terrie',
                'miniature american eskimo dog',
                'puli',
                'shiba inu',
                'skye terrier',
                'spinone italiano',
                'swedish vallhund',
                'tibetan spaniel',
                'toy fox terrier',
                'toy manchester terrier',
                'welsh terrier']   
    return className

def return_top_indices(predict_vec):
  indices=heapq.nlargest(3, range(len(predict_vec)), key=predict_vec.__getitem__)
  probs=heapq.nlargest(3, predict_vec)
  top_indices=[]
  top_probs=[]
  for i in range(0,3):
    if probs[i]<=0:
      break
    top_indices.append(indices[i])
    top_probs.append(probs[i])
  return top_indices,top_probs

               
def Main_Processing(image):
    try:
        #npimg = np.fromstring(image, np.uint8)
        #img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
        img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        predicted_vector=predict_breed(img)
        
        ##tf.keras.backend.clear_session()
        #index=np.argmax(predicted_vector)
        #breed=dog_names[index] 
        top_indices,top_probs=return_top_indices(predicted_vector)
        dog_names=return_dog_names()
        breeds=[dog_names[i] for i in top_indices]
        #prob=predicted_vector[index]
        dic={"api_status":200,"breed":str(breeds),"prob":str(top_probs)}
        return dic
    except Exception as e:
        print(e)
        return {"api_status":400,"msg":"can not handle this image","problem":str(e)}

def Main_Processing_For_Identification(img):
    try:
        # npimg = np.fromstring(image, np.uint8)
        # img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        predicted_vector=predict_breed(img)
        
        ##tf.keras.backend.clear_session()
        #index=np.argmax(predicted_vector)
        #breed=dog_names[index] 
        top_indices,top_probs=return_top_indices(predicted_vector)
        dog_names=return_dog_names()
        breeds=[dog_names[i] for i in top_indices]
        #prob=predicted_vector[index]
        dic={"breed":str(breeds),"prob":str(top_probs)}
        return dic
    except Exception as e:
        print(e)
        return {"msg":"can not handle this image","problem":str(e)}