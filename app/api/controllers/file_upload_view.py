import uuid
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from ..utils.file_serializer import FileSerializer
import os
import base64

@api_view(['POST'])
def file_upload(request):
    serializer = FileSerializer(data=request.data)

    if serializer.is_valid():
        file = serializer.validated_data['file']
        file_name = file.name
        file_path = os.path.join('uploads', file_name)

        print(file_path)

        with open(file_path, 'wb') as f:
            f.write(file.read())
        return Response(status=status.HTTP_201_CREATED)

    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['POST'])
def upload_architecture(request):
    str_base64 = request.data['base64']

    decoded_file = base64.b64decode(str_base64)

    # Generate file name
    file_extension = 'py' # default file extension
    file_name = f'architecture.{file_extension}'

    # Create uploads directory if it doesn't exist
    if not os.path.exists('api/utils'):
        os.makedirs('api/utils')

    # Write decoded file to uploads directory
    file_path = os.path.join('api/utils', file_name)
    with open(file_path, 'wb') as f:
        f.write(decoded_file)

    # Create file URL for response
    file_url = request.build_absolute_uri('/')[:-1] + f'/api/utils/{file_name}'

    # Return response
    return Response({'file_name': file_name, 'file_url': file_url}, status=status.HTTP_201_CREATED)

@api_view(['POST'])
def upload_model(request):
    str_base64 = request.data['base64']

    decoded_file = base64.b64decode(str_base64)

    # Generate file name
    file_extension = 'h5' # default file extension
    file_name = f'siamese_model.{file_extension}'

    # Create uploads directory if it doesn't exist
    if not os.path.exists('api/models'):
        os.makedirs('api/models')

    # Write decoded file to uploads directory
    file_path = os.path.join('api/models', file_name)
    with open(file_path, 'wb') as f:
        f.write(decoded_file)

    # Create file URL for response
    file_url = request.build_absolute_uri('/')[:-1] + f'/api/models/{file_name}'

    # Return response
    return Response({'file_name': file_name, 'file_url': file_url}, status=status.HTTP_201_CREATED)
       