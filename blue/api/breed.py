import json
import requests
from requests import Request, Session
from PIL import Image
import PIL
import numpy as np
from io import  BytesIO
import cv2
import os

dirname = os.path.dirname(os.path.abspath(__file__))
dirname_list = dirname.split("/")[:-1]
dirname = "/".join(dirname_list)
dirname = dirname + "/api/"

def breed_processing(image_url):
    
    print("start breed")
    print(image_url)
   
    headers = {
                'Accept': 'application/json'
            }
    files = {
            'image_url': image_url
            }
    api_url = 'http://localhost:5000/api/dog_breeds'
    normal_multipart_req = Request('POST', api_url, json=files).prepare()
    request = normal_multipart_req
    print(request)
    print(type(request))
    s = Session()
    response = s.send(request)
    print("REsponse",response)

    if response.status_code != 200:
        ret = [0,"Didnt get any response"]
        print(response.status_code, response.text)
        print(ret)
        return ret

    else:
        try:
            #print(response.status_code, response.text)
            response_text = response.text
            response_dict = json.loads(response_text)
            print(response_dict)
            print(type(response_dict))
            ret = ["Not_Exception",response_dict]
            print(ret)
            return ret

        except Exception as e:
            ret = ["Exception",e]
            return ret    


#breed_processing('https://ladoo.petmypal.biz/upload/photos/2020/10/xsgzB7PbB1LN9fqbSwcL_14_a5f92b6fe209c18f225f2eb973090c6d.png')
