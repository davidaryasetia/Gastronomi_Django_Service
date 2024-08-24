from django.http import JsonResponse

def HelloWorld(request):
    data = {
        "message" : "Hello World"
    }
    return JsonResponse(data)