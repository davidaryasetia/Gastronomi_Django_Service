from rest_framework.response import Response
from rest_framework.decorators import api_view
from datetime import datetime
from PIL import Image
from io import BytesIO
from numpy import asarray
from ..utils.architecture import siamese_architecture
from django.views.decorators.csrf import csrf_exempt
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

# Test Request Post IMAGE


@api_view(['POST'])
def predict_with_data(request):

    # menggunakan data contoh statis 
    foods = [
        {
            'picture' : "C:\\Users\\evane\\Documents\\Asset Project\\Food\\Ayam Taliwang\\Ayam Taliwang 1.jpg",
            'name' : 'Ayam Taliwang', 
            'foodCode' : 'F001', 
            '_id' : '1'
        }, 
        {
            'picture' : "C:\\Users\\evane\\Documents\\Asset Project\\Food\\Ayam Rarang\\ayam rarang1.jpg", 
            'name' : 'Ayam Rarang', 
            'foodCode' : 'F002',
            '_id' : 'F002',
        }, 
        {
            'picture' : "C:\\Users\\evane\\Documents\\Asset Project\\Food\Bebalung\\Bebalung1.jpeg", 
            'name' : 'Bebalung', 
            'foodCode' : 'F003',
            '_id' : 'F002',
        },
        {
            'picture' : "C:\\Users\\evane\\Documents\\Asset Project\\Food\\Plecik Kangkung\\plecik kangkung1.jpeg", 
            'name' : 'Plecing Kangkung', 
            'foodCode' : 'F004',
            '_id' : 'F004',
        }, 
    ]

    # Memuat model Siamese untuk melakukan predisi
    siamese_model = siamese_architecture()
    cwd = os.getcwd()  
    siamese_model.load_weights(cwd+"/app/api/models/siamese_model.h5")

    # Kondisi jika kita tidak memasukkan sebuah image 
    if 'query' not in request.FILES:
        return Response({'error' : 'No query image uploaded'}, status=400)
    
    # membaca gambar dari parameter query
    query_image_file = request.FILES['query']
    imgQuery = read_image(query_image_file)

    images = []
    labels = []

    for food in foods:
        # images.append(food['picture'][0])
        image_path = food['picture']
        img = read_image(image_path)
        with open(image_path, 'rb') as img_file:
            img = read_image(img_file)
            images.append(img)

        labels.append({ # label image
            'name' : food['name'],
            'foodCode' : food['foodCode'],
            '_id' : food['_id']
        })

    # Membuat array gambar untuk input model
    imgArr1 = np.array([imgQuery] * len(images)) # Menggandakan gambar query untuk setiap gambar referensi
    imgArr2 = np.array(images) #gambar refrensi

    results = siamese_model.predict([imgArr1,imgArr2])

    predicts = []

    # Iterasi hasil prediksi dan menyimpannya bersama label 
    for i in range(len(results)):
        predicts.append({
            'label': labels[i],
            'predict':float(results[i])
        })

    # Mengambil hasil prediksi tertinggi berdasarkan nilai
    highest_predict = max(predicts, key=lambda x: x['predict'])

    return Response(highest_predict)

def read_image(image_file):
    image = Image.open(image_file)
    image = image.resize(IMAGE_SHAPE)
    img_arr = np.asarray(image)
    return img_arr.astype('float32')



@api_view(['POST'])
def multi_predict(request):
    siamese_model = siamese_architecture()
    cwd = os.getcwd()  
    siamese_model.load_weights(cwd+"/app/api/models/siamese_model.h5")
    
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
    siamese_model.load_weights(cwd+"/app/api/models/siamese_model.h5")

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
