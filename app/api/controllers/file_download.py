import os
from django.http import HttpResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

@api_view(['GET'])
def file_download(request, file_name):

    if not os.path.exists('api/utils'):
        return Response(status=status.HTTP_400_BAD_REQUEST)

    file_path = os.path.join('api/utils', file_name)

    # Open the file in binary mode
    with open(file_path, 'rb') as f:
        response = HttpResponse(f.read(), content_type='application/octet-stream')
        response['Content-Disposition'] = f'attachment; filename="{file_name}"'
        return response
