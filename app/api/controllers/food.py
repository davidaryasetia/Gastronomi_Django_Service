from rest_framework.response import Response
from rest_framework.decorators import api_view
from datetime import datetime
from PIL import Image
from io import BytesIO
from numpy import asarray
from ..utils.architecture import siamese_architecture
import os
import base64
import numpy as np
import requests
import dotenv
dotenv.load_dotenv()
FOOD_HOST = os.getenv('FOOD_HOST')
HOST = os.getenv('HOST')


IMAGE_SHAPE = (224,224)

@api_view(['GET'])
def hello(request):
    return Response({'say':'hello'})


@api_view(['POST'])
def predict_with_data(request):
    foods = requests.get(FOOD_HOST)
    siamese_model = siamese_architecture()
    cwd = os.getcwd()  
    siamese_model.load_weights(cwd+"/api/models/siamese_model.h5")
    
    imgQuery = request.data['query']
    images = []
    labels = []

    for food in foods.json():
        # for picture in food['picture']:
        #     images.append(picture)
        #     labels.append({
        #         'name' : food['name'],
        #         'foodCode' : food['foodCode'],
        #         '_id' : food['_id']
        #     })
        images.append(food['picture'][0])
        labels.append({
            'name' : food['name'],
            'foodCode' : food['foodCode'],
            '_id' : food['_id']
        })

    imgArr1 = get_duplicate_array_image(imgQuery, len(images))
    imgArr2 = get_multi_array_image_link(images)

    results = siamese_model.predict([imgArr1,imgArr2])

    predicts = []

    for i in range(len(results)):
        predicts.append({
            'label': labels[i],
            'predict': results[i]
        })

    return Response(predicts)

@api_view(['POST'])
def multi_predict(request):
    siamese_model = siamese_architecture()
    cwd = os.getcwd()  
    siamese_model.load_weights(cwd+"/api/models/siamese_model.h5")
    
    imgQuery = request.data['query']
    images = request.data['images']

    imgArr1 = get_duplicate_array_image(imgQuery, len(images))
    imgArr2 = get_multi_array_image(images)

    result = siamese_model.predict([imgArr1,imgArr2])

    return Response(result)

@api_view(['POST'])
def predict(request):
    siamese_model = siamese_architecture()
    cwd = os.getcwd()  
    siamese_model.load_weights(cwd+"/api/models/siamese_model.h5")

    imgArr1 = get_array_image(request.data['image1'])
    imgArr2 = get_array_image(request.data['image2'])

    result = siamese_model.predict([imgArr1,imgArr2])

    return Response({'predict':result[0][0]})

def get_multi_array_image_link(multi_images):
    x = []
    for base in multi_images:
        response = requests.get(HOST+base)
        image = Image.open(BytesIO(response.content))
        image = image.resize(IMAGE_SHAPE)
        imgArr = asarray(image)
        x.append(imgArr)
    return np.array(x).astype('float32')  

def get_multi_array_image(multi_base64):
    x = []
    for base in multi_base64:
        image = Image.open(BytesIO(base64.b64decode(base)))
        image = image.resize(IMAGE_SHAPE)
        imgArr = asarray(image)
        x.append(imgArr)
    return np.array(x).astype('float32')  

def get_duplicate_array_image(str_base64, num):
    x = []
    for k in range(num):
        image = Image.open(BytesIO(base64.b64decode(str_base64)))
        image = image.resize(IMAGE_SHAPE)
        imgArr = asarray(image)
        x.append(imgArr)
    return np.array(x).astype('float32')  

def get_array_image(str_base64):
    x = []
    image = Image.open(BytesIO(base64.b64decode(str_base64)))
    image = image.resize(IMAGE_SHAPE)
    imgArr = asarray(image)
    x.append(imgArr)

    return np.array(x).astype('float32')   
