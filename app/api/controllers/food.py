from rest_framework.response import Response
from rest_framework.decorators import api_view
from PIL import Image
from io import BytesIO
from ..utils.architecture import siamese_architecture
import os
import numpy as np
import requests


@api_view(['POST'])
def predict_with_data(request):
    
    url = 'http://gastronomy-backend.sindika.co.id/api/food'
    response = requests.get(url)
    api_data = response.json()
    foods = api_data['data']

    # Memuat model Siamese untuk melakukan predisi
    siamese_model = siamese_architecture()
    cwd = os.getcwd()  
    siamese_model.load_weights(cwd+"/app/api/models/siamese_model.h5")

    if 'query' not in request.FILES:
        return Response({'error' : 'No query image uploaded'}, status=400)
    
    # membaca gambar dari parameter query
    query_image_file = request.FILES['query']
    imgQuery = read_image(query_image_file)

    images = []
    labels = []

    for food in foods:
        food_id = food['food_id']
        photo_path = food['photo_path'] 
        name = food['name']
        category = food['category']
        description = food['description']
        story_historical_food = food['food_historical']
        ingredients = food['ingredients']
        directions = food['directions']
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
            'directions' : directions, 
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
