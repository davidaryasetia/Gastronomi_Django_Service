from django.http import JsonResponse
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


def HelloWorld(request):
    data = {
        "message" : "Hello Guys"
    }
    return JsonResponse(data)


# @api_view(['POST'])
# def predict_with_data(request):
#     foods = requests.get(FOOD_HOST) #base 64 image
#     siamese_model = siamese_architecture()
#     cwd = os.getcwd()  
#     siamese_model.load_weights(cwd+"/app/api/models/siamese_model.h5")
    
#     imgQuery = request.data['query']
#     images = []
#     labels = []

#     for food in foods.json():
#         # for picture in food['picture']:
#         #     images.append(picture)
#         #     labels.append({
#         #         'name' : food['name'],
#         #         'foodCode' : food['foodCode'],
#         #         '_id' : food['_id']
#         #     })
#         images.append(food['picture'][0])
#         labels.append({
#             'name' : food['name'],
#             'foodCode' : food['foodCode'],
#             '_id' : food['_id']
#         })

#     imgArr1 = get_duplicate_array_image(imgQuery, len(images))
#     imgArr2 = get_multi_array_image_link(images)

#     results = siamese_model.predict([imgArr1,imgArr2])

#     predicts = []

#     for i in range(len(results)):
#         predicts.append({
#             'label': labels[i],
#             'predict': results[i]
#         })

#     return Response(predicts)