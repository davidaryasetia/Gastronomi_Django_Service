import requests
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods


# Testing Request Data Food From API Laravel Backend 
@require_http_methods(["GET"])
def food_data(request):
    api_url = 'http://gastronomy-backend.sindika.co.id/api/food/1'

    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse(data)