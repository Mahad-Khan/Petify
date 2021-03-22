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

dirname = os.path.dirname(os.path.abspath(__file__))
dirname_list = dirname.split("/")[:-1]
dirname = "/".join(dirname_list)
print(dirname)
path = dirname + "/api"
sys.path.append(path)
cropimage_path = path + "/temp/"
register_path=path+"/registered"
sys.path.append(path)
db_path = dirname + "/MissingDB"
print(db_path)
sys.path.append(db_path)

from Functions import Main_Processing
from Functions import Main_Processing_For_Identification
from yolo_object_detection import yolo_return_names
# from Upload import upload_file
# from Upload import upload_file_guest
# from lost import lost_dog_list
# from download_guest import dosimage_guest
# from download import dosimage
from breed import breed_processing
from image_processing import image_matching 


#from blue import app
import logging

import tracemalloc
tracemalloc.start()

mod = Blueprint('api',__name__)
api = Api(mod)

#database
db = SQLAlchemy()

logging.basicConfig(filename='demo.log',level=logging.DEBUG,
format="%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s")



# class user(db.Model):
#     __tablename__ = "AI_Pet_table"
#     id = db.Column(db.String(50), primary_key=True) #pet id
#     breed = db.Column(db.String(50))                #pet breed
#     def __repr__(self):
#         return('REGISTERED PET '+str(self.id)+' '+str(self.breed))

class Pet(db.Model):
    __tablename__ = "Pet_Table"
    id = db.Column(db.String(50), primary_key=True) #pet id
    breed = db.Column(db.String(50))                #pet breed
    url = db.Column(db.Text())
    lost = db.Column(db.String(1))
    def __repr__(self):
        return('REGISTERED PET '+str(self.id)+' '+str(self.breed)+' '+str(self.url)+' '+str(self.lost))
        #return [str(self.id),str(self.breed),str(self.url),str(self.lost)]


class Found(db.Model):
    __tablename__ = "Found_Table"
    gid = db.Column(db.String(50), primary_key=True) #guest id
    breed = db.Column(db.String(50))                 #pet breed
    url = db.Column(db.Text())                  #pet url
    def __repr__(self):
        return('REGISTERED PET '+str(self.gid)+' '+str(self.breed)+' '+str(self.url))


class guest(db.Model):
    __tablename__ = "AI_Guest_DB"
    gid = db.Column(db.String(50), primary_key=True) #guest_id
    geq = db.Column(db.Float()) # dog_equation
    brd = db.Column(db.String(50)) #breed
    #glog = db.Column(db.Float())
    #glat = db.Column(db.Float())
    
    def __repr__(self):
        #return('GUEST PET'+str(self.gid)+' '+str(self.geq)+' '+str(self.brd))
        return('GUEST PET'+str(self.gid)+' '+str(self.geq))

class lost_matching(db.Model):
    __tablename__ = "Lost_Matching_DB"
    reg_id = db.Column(db.String(50), primary_key=True)
    guest_id = db.Column(ScalarListType())
    guest_target = db.Column(ScalarListType())
    def __repr__(self):
        return('LOST PET '+str(self.reg_id)+' '+str(self.guest_id)+' '+str(self.guest_target))


def string_to_list(st):
    new_st = st.split("[")[1]
    new_st_upd = new_st.split("]")[0]
    comma_values = new_st_upd.split(",")
    my_list = list()
    for i in range(len(comma_values)):
        value = comma_values[i].split("'")
        string_breed = value[1]
        my_list.append(string_breed)
        
    return my_list


class_num = {'affenpinscher':1,#null
                'afghan_hound':2,
                'airedale_terrier':3,
                'akita':4,
                'alaskan_malamute':5,
                'american_eskimo_dog':6,#miss
                'american_foxhound':7,
                'american_staffordshire_terrier':8,
                'american_water_spaniel':9,
                'anatolian_shepherd_dog':10,
                'australian_cattle_dog':11,
                'australian_shepherd':12,
                'australian_terrier':13,
                'basenji':14,
                'basset_hound':15,
                'beagle':16,
                'bearded_collie':17,
                'beauceron':18,
                'bedlington_terrier':19,
                'belgian_malinois':20,
                'belgian_sheepdog':21,
                'belgian_tervuren':22,
                'bernese_mountain_dog':23,
                'bichon_frise':24,
                'black_and_tan_coonhound':25,
                'black_russian_terrier':26,
                'bloodhound':27,
                'bluetick_coonhound':28,#miss
                'border_collie':29,
                'border_terrier':30,
                'borzoi':31,
                'boston_terrier':32,
                'bouvier_des_flandres':33,
                'boxer':34,
                'boykin_spaniel':35,#miss
                'briard':36,
                'brittany':37,
                'brussels_griffon':38,
                'bull_terrier':39,
                'bulldog':40,
                'bullmastiff':41,
                'cairn_terrier':42,
                'canaan_dog':43,
                'cane_corso':44,#miss
                'cardigan_welsh_corgi':45,
                'cavalier_king_charles_spaniel':46,
                'chesapeake_bay_retriever':47,
                'chihuahua':48,
                'chinese_crested':49,
                'chinese_shar-pei':50,
                'chow_chow':51,
                'clumber_spaniel':52,
                'cocker_spaniel':53,
                'collie':54,
                'curly-coated_retriever':55,
                'dachshund':56,
                'dalmatian':57,
                'dandie_dinmont_terrier':58,
                'doberman_pinscher':59,
                'dogue_de_bordeaux':60,
                'english_cocker_spaniel':61,
                'english_setter':62,
                'english_springer_spaniel':63,
                'english_toy_spaniel':64,
                'entlebucher_mountain_dog':65,#miss
                'field_spaniel':66,
                'finnish_spitz':67,
                'flat-coated_retriever':68,
                'french_bulldog':69,
                'german_pinscher':70,
                'german_shepherd_dog':71,
                'german_shorthaired_pointer':72,
                'german_wirehaired_pointer':73,
                'giant_schnauzer':74,
                'glen_of_imaal_terrier':75,
                'golden_retriever':76,
                'gordon_setter':77,
                'great_dane':78,
                'great_pyrenees':79,
                'greater_swiss_mountain_dog':80,
                'greyhound':81,
                'havanese':82,
                'ibizan_hound':83,
                'icelandic_sheepdog':84,#miss
                'irish_red_and_white_setter':85,#miss
                'irish_setter':86,
                'irish_terrier':87,
                'irish_water_spaniel':88,
                'irish_wolfhound':89,
                'italian_greyhound':90,
                'japanese_chin':91,
                'keeshond':92,
                'kerry_blue_terrier':93,
                'komondor':94,
                'kuvasz':95,
                'labrador_retriever':96,
                'lakeland_terrier':97,
                'leonberger':98,#miss
                'lhasa_apso':99,
                'lowchen':100,
                'maltese':101,#miss
                'manchester_terrier':102,#miss
                'mastiff':103,
                'miniature_schnauzer':104,
                'neapolitan_mastiff':105,
                'newfoundland':106,
                'norfolk_terrier':107,
                'norwegian_buhund':108,#miss 
                'norwegian_elkhound':109,
                'norwegian_lundehund':110,#miss
                'norwich_terrier':111,
                'nova_scotia_duck_tolling_retriever':112,#null 
                'old_english_sheepdog':113,#miss
                'otterhound':114,
                'papillon':115,
                'parson_russell_terrier':116,
                'pekingese':117,
                'pembroke_welsh_corgi':118,
                'petit_basset_griffon_vendeen':119,
                'pharaoh_hound':120,
                'plott':121,
                'pointer':122,
                'pomeranian':123,
                'poodle':124,#miss
                'portuguese_water_dog':125,
                'saint_bernard':126,
                'silky_terrier':127,
                'smooth_fox_terrier':128,
                'tibetan_mastiff':129,
                'welsh_springer_spaniel':130,
                'wirehaired_pointing_griffon':131,#d
                'xoloitzcuintli':132,#miss
                'yorkshire_terrier':133,
                'shetland_sheepdog':134,
                'english_foxhound':135,
                'african_hunting_dog':136,
                'dhole':137,
                'dingo':138,
                'mexican_hairless':139,
                'standard_poodle':140,#d
                'miniature_poodle':141,
                'toy_poodle':142,
                'brabancon_griffon':143,
                'samoyed':144,
                'pug':145,
                'malamute':146,
                'eskimo_dog':147,
                'entleBucher':148,
                'appenzeller':149,
                'miniature_pinscher':150,
                'rottweiler':151,
                'kelpie':152,
                'malinois':153,
                'groenendael':154,
                'schipperke':155,
                'siberian_husky':156,
                'sussex_spaniel':157,
                'vizsla':158,
                'west_Highland_white_terrier':159,
                'scotch_terrier':160,
                'sealyham_terrier':161,
                'irish_terrier':162,
                'shih-Tzu':163,
                'japanese_spaniel':164,
                'redbone':165,
                'walker_hound':166,
                'wire-haired_fox_terrier':167,
                'whippet':168,
                'weimaraner':169,
                'soft-coated_wheaten_terrier':170,
                'staffordshire_bullterrier':171,
                'scottish_deerhound':172,
                'saluki':173,
                'blenheim_spaniel':174,
                'toy_terrier':175,
                'rhodesian_ridgeback':176,
                'standard_schnauzer':177,
                'tibetan_terrier':178,
                'miniature poodle':179,
                'harrier':180,
                'jack russel terrier':181,
                'polish lowland sheepdog':182,
                'dogo argentino':183,
                'miniature bull terrie':184,
                'miniature american eskimo dog':185,
                'puli':186,
                'shiba inu':187,
                'skye terrier':188,
                'spinone italiano':189,
                'swedish vallhund':190,
                'tibetan spaniel':191,
                'toy fox terrier':192,
                'toy manchester terrier':193,
                'welsh terrier':194
}

class Dog_Breeds(Resource):
    def post(self):
       
        try:
            ##tf.keras.backend.clear_session()
            
            #app.logger.info('Processing default request')
            print("imager")
            #image = request.files['image'].read()
            postedData=request.get_json()
            print(postedData)
            image_url=postedData['image_url']
            response=requests.get(image_url)
            img=Image.open(BytesIO(response.content))
            # valid=validators.url(image_url)
            # print(valid)
            # response=requests.get(image_url)
            # img = Image.open(BytesIO(response.content))
            
            imgarr = np.array(img) 
            l=yolo_return_names(imgarr)
            
            print("sdfsd")
            if "dog" in l:

                print("-----------------------HITT------------------")
                # ... run your application ...

                ret=Main_Processing(imgarr)
                for stat in top_stats[:15]:
                    print(stat)
                tf.keras.backend.clear_session()
                gc.collect()
                return jsonify(ret)
            else:
                tf.keras.backend.clear_session()
                gc.collect()
                return jsonify({"api_status":400,"msg":"Dog image is not present"})
        except Exception as e:
            return jsonify({"api_status":400,"msg":"Can not handle this image","problem":e})

class Register(Resource):
    def post(self):
        try:
            #app.logger.info('Processing default request')
            print("-----------------------HITT------------------")
            postedData=request.get_json()
            print(postedData)
            image_url=postedData['image_url']
            PUID=postedData['puid']
            print(PUID)
            breed=postedData['breed']
            #puid_in_db=user.query.filter(user.id.like(PUID))
            #print(puid_in_db)
            exists = bool(Pet.query.filter_by(id=PUID).first())
            print(exists)
            if exists:
                return jsonify({"api_status":400,"msg":"Registration ID must be unique"})   
            #image_url = 'https://upload.wikimedia.org/wikipedia/commons/4/47/American_Eskimo_Dog.jpg'
            valid=validators.url(image_url)
            if valid==True:
                print(image_url)
                print(PUID)
                print(breed)
                response = requests.get(image_url)
                img = Image.open(BytesIO(response.content))
                imgarr = np.array(img)
                print("url to image")
                #rgb_image = cv2.cvtColor(imgarr,cv2.COLOR_BGR2RGB)
                if breed == None or breed == "":
                    #Find Breed
                    dic=Main_Processing_For_Identification(imgarr)
                    print(dic)
                    breed_list_add = dic["breed"]
                    print("breed list: ",breed_list_add)
                    breed_list=string_to_list(breed_list_add)
                    find_breed = breed_list[0]
                    print("breed:",find_breed)
                else:
                    #Find Breed
                    find_breed = breed
                    print("breed:",find_breed)

                #breed_path=register_path+"/"+find_breed.capitalize()+"/"
                #print(breed_path)
                #cv2.imwrite(os.path.join(breed_path,str(PUID)+'.jpg'),rgb_image) 
                pet=Pet(id=PUID,breed=find_breed,url=image_url,lost="f")
                db.session.add(pet)
                print(pet)
                db.session.commit()
                return jsonify({"api_status":200,"msg":"Successfully Registered Pet"})
            if valid == False:
                ret = {"api_status":400,"msg":"Enter Valid URL","problem":"Invalid URL"}
                return jsonify(ret)
            tf.keras.backend.clear_session()
            gc.collect()
    
        except Exception as e:
            ret = {"api_status":301,"msg":"Unsuccessful Registration","problem":e}
            return jsonify(ret)


class Breed_Dict(Resource):
    def get(self):
        try:
            print("-----------------------HITT------------------")
            print(class_num)   

            pickle.dump(class_num, open("Breed_dict.p", "wb")) 

            return jsonify(class_num)
        except Exception as e:
            ret = {"api_status":400,"msg":"Unsuccessful Attempt","problem":e}
            return jsonify(ret)

class Lost_list(Resource):
    def get(self):
        try:
            print("-----------------------HITT------------------")
            
            rows = lost_matching.query.with_entities(lost_matching.reg_id, lost_matching.guest_id,lost_matching.guest_target).all()
            print(rows)
            record={}

            for each_record in rows:
                print(each_record)
                guest_ids = each_record[1]
                #print(guest_ids)
                guest_target = each_record[2]
                #print(guest_target)
                for x in range(len(guest_ids)):
                    print(x)
                    record[each_record[0]]=[guest_ids[x],guest_target[x]]

            print(record)    

            return jsonify(record)

        except Exception as e:
            ret = {"api_status":400,"msg":"Unsuccessful Attempt","problem":e}
            return jsonify(ret)



class Extract_Features(Resource):
    def post(self):
        try:
            #app.logger.info('Processing default request')
            image = request.files['image'].read()
            # breed_list = request.form['breeds']
            # print(breed_list)
            
            print("-----------------------HITT------------------")
            # ... run your application ...

            ret= Feature_Match(image,db_path,detector,predictor)

            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            print("[ Top 15 ]")
            for stat in top_stats[:15]:
                print(stat)
            tf.keras.backend.clear_session()
            gc.collect()
            return jsonify(ret)

        except Exception as e:
            ret = {"status":301,"msg":"Picture quality is not good","problem":e}
            gc.collect()
            return jsonify(ret)

class Add_Dog_In_DB(Resource):
    def post(self):
        try:
            #app.logger.info('Processing default request')
            image = request.files.get('image', '')
            print("type",type(image))
            filename = image.filename
            #breed = request.form['breed']
            #dog_id = request.form['dog_id']
            
            #filename = image.filename
            # breed_path = db_path + "/" + breed
            # if not os.path.exists(breed_path):
            #     os.makedirs(breed_path)

            try:
                print("-----------------------HITT------------------")
                # ... run your application ...

                #msg = Update_Landmarks(image,breed,filename,db_path,detector,predictor)
                msg = Update_Landmarks(image,filename,db_path,detector,predictor)
                snapshot = tracemalloc.take_snapshot()
                top_stats = snapshot.statistics('lineno')
                print("[ Top 15 ]")
                for stat in top_stats[:15]:
                    print(stat)
                tf.keras.backend.clear_session()
                gc.collect()

                if msg == "Successfully Updated":
                    status = 200
                else:
                    status = 301

                ret = {"status":status,"msg":msg}
                return jsonify(ret)
            
            except Exception as e:
                ret = {"status":301,"msg":"Picture quality is not good","problem":e}
                return jsonify(ret)
        except Exception as e:
            ret = {"status":301,"msg":"Cannot add this image in DB","problem":e}
            return jsonify(ret)


class Image_Processing(Resource):
    def post(self):
        try:
            print("-----------------------HITT------------------")
            Data=request.get_json() 
            query_url=Data["query_url"]
            urls=Data["urls"]
            urls,costs=image_matching(query_url,urls)
            ret={"status":200,"urls":urls,"costs":costs}
            return jsonify(ret)
        except Exception as e:
            ret = {"api_status":400,"msg":"Unsuccessful Attempt","problem":e}
            return jsonify(ret)

class Lost(Resource):
    def post(self):
        try:
            print("-----------------------HITT------------------")
            Data=request.get_json() 
            dog_id=Data["DID"]  ##must be unique
            print(dog_id)
            #get url from DB using DID
            exists = bool(Pet.query.filter_by(id=dog_id).first()) ## work on bool
            print(exists)
            if exists:
                exists=Pet.query.filter_by(id=dog_id).first()
                print(exists)
                exists.lost='t'
                db.session.commit()
                print("after commit:",exists)
                lost_dog_url=exists.url
                lost_dog_breed=exists.breed
                exists = bool(Found.query.filter_by(breed=lost_dog_breed).first())
                print(exists)
                if exists:
                    found_db =Found.query.filter(Pet.breed == lost_dog_breed).all()
                    print(found_db)
                    urls = [row.url for row in found_db]
                    print(urls)
                    #get urls from found DB filter by breed
                    #db.session.delete(row)
                    #db.session.commit()
                    if urls == None:
                        ret={"status":200,"msg":"url is none"}
                        return jsonify(ret)
                    else:
                        #call image matching func
                        urls,costs=image_matching(lost_dog_url,urls)    #return most similar url to the user
                        ret={"status":200,"urls":urls,"costs":costs}
                        return jsonify(ret)
                else:
                    ret={"status":200,"msg":"no same breed in found table"}
                    return jsonify(ret)
            else:
                return jsonify({"api_status":400,"msg":"Dog ID is not registered"})
            
        except Exception as e:
            ret = {"api_status":400,"msg":"Unsuccessful Attempt","problem":e}
            return jsonify(ret)

class Found_API(Resource):
    def post(self):
        try:
            print("-----------------------HITT------------------")
            # num_rows_deleted = Found.query.delete()
            # print(num_rows_deleted)
            Data=request.get_json() 
            guest_id=Data["guest_ID"] ##must be unique
            query_url=Data["url"]
            print(guest_id,query_url)
            exists = bool(Found.query.filter_by(gid=guest_id).first())
            print(exists)
            if exists:
                return jsonify({"api_status":400,"msg":"Guest ID must be unique"})   
            valid=validators.url(query_url)
            print(valid)
            if valid==True:
                response = requests.get(query_url)
                img = Image.open(BytesIO(response.content))
                imgarr = np.array(img)
                # rgb_image = cv2.cvtColor(imgarr,cv2.COLOR_BGR2RGB)
                dic=Main_Processing_For_Identification(imgarr)
                print(dic)
                breed_list_add = dic["breed"]
                print("breed list: ",breed_list_add)
                breed_list=string_to_list(breed_list_add)
                find_breed = breed_list[0]
                print("breed:",find_breed)
                ##insertion in found table
                guest_row=Found(gid=guest_id,breed=find_breed,url=query_url)
                db.session.add(guest_row)
                print(guest_row)
                db.session.commit()
                ##getting results from pet_reg table,filter by breed and lost
                exists=bool(Pet.query.filter_by(breed=find_breed,lost='t').first())
                print(exists)
                if exists: ##notify the user 
                    pet_reg_data=Pet.query.filter(Pet.breed==find_breed,Pet.lost=='t').all()
                    print(pet_reg_data)
                    urls = [row.url for row in pet_reg_data]
                    #pet_ids=[row.id for row in pet_reg_data]
                    print(urls)
                    urls,costs=image_matching(query_url,urls)
                    top_url=urls[0]
                    print(top_url)
                    user_row=Pet.query.filter_by(url=top_url).first()
                    user_id=user_row.id
                    print(user_id)
                    return jsonify({"api_status":200,"top_url":top_url,"user":user_id,"guest_id":guest_id})
                else:##won't notify the user
                    return jsonify({"api_status":200,"msg":"no match in pet table"})
            else:
                ret = {"api_status":400,"msg":"Enter Valid URL","problem":"Invalid URL"}
                return jsonify(ret)

            urls,costs=image_matching(query_url,urls)
            ret={"status":200,"urls":urls,"costs":costs}
            return jsonify(ret)
        except Exception as e:
            ret = {"api_status":400,"msg":"Unsuccessful Attempt","problem":e}
            return jsonify(ret)

class Yes_Found(Resource):
    def post(self):
        try:
            print("-----------------------HITT------------------")
            Data=request.get_json() 
            pet_id=Data["pet_id"] 
            guest_id=Data['guest_id']
            exists=bool(Pet.query.filter_by(id=pet_id).first())
            print(exists)
            if not(exists):
                ret={"status":200,"msg":"pet id not found"}
                return jsonify(ret)
            exists=bool(Found.query.filter_by(gid=guest_id).first())
            if not(exists):
                ret={"status":200,"msg":"guest id not found"}
                return jsonify(ret)
            db.session.query(Found).filter(Found.gid==guest_id).delete()
            db.session.commit()
            pet_row=Pet.query.filter_by(id=pet_id).first()
            pet_row.lost='f'
            db.session.commit()
            ret={"status":200,"msg":"tables have been updated"}
            return jsonify(ret)
        except Exception as e:
            ret = {"api_status":400,"msg":"Unsuccessful Attempt","problem":e}
            return jsonify(ret)


api.add_resource(Register,"/register")
api.add_resource(Lost,"/lost")
api.add_resource(Found_API,"/found")
api.add_resource(Yes_Found,"/yes_found")

api.add_resource(Dog_Breeds,"/dog_breeds")
# api.add_resource(Guest_find,"/guest_dog")
# api.add_resource(Breed_Dict,"/breed_data")
# api.add_resource(Lost_list,"/lost_list")
# api.add_resource(Guest_IDs,"/guest_id")
# api.add_resource(Guest_Matching,"/guest_matching")

#api.add_resource(Dog_check,"/check")
#api.add_resource(Hit_Web,"/hit")
#api.add_resource(Hit_Web_Again,"/hit_again")
#api.add_resource(Breed,"/dog_seq")
# api.add_resource(Extract_Features,"/features")
# api.add_resource(Add_Dog_In_DB,"/upload")

