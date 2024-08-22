from django.http import JsonResponse

def simple_get_request(request):
    data = {
        "message": "Hello World"
    }
    return JsonResponse(data)
