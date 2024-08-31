from rest_framework.response import Response
from rest_framework.decorators import api_view
from django.http import JsonResponse
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
    url = 'http://gastronomy-backend.sindika.co.id/api/food'
    try:
        response = requests.get(url)
        response.raise_for_status()
        api_data = response.json()

        if 'data' not in api_data: 
            return JsonResponse({'error' : 'Invalid data format from API'}, status=500)
        
        foods = api_data['data']

    except requests.exceptions.RequestException as e:
        return JsonResponse({'error' : str(e)}, status= 500)

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
        food_id = food['food_id']
        photo_path = food['photo_path'] 
        name = food['name']
        category = food['category']
        description = food['description']
        story_historical_food = food['food_historical']
        ingredients = food['ingredients']
        url_youtube = food['url_youtube']
        nutrition = food['nutrition']

            
        img = read_image(photo_path)
        images.append(img)

        labels.append({
            'food_id' : food_id, 
            'photo_path' : photo_path, 
            'name' : name, 
            'category' : category, 
            'description' : description, 
            'story_historical_food' : story_historical_food, 
            'ingredients' : ingredients, 
            'url_youtube' : url_youtube, 
            'nutrition' : nutrition, 
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

# Fungsi read_image file 
def read_image(image_input):
    if isinstance(image_input, str) and (image_input.startswith('http://') or image_input.startswith('https://')):
        response = requests.get(image_input)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
    else: 
        image = Image.open(image_input)

    image = image.resize((224, 224))
    img_arr = np.asarray(image)
    return img_arr.astype('float32')
