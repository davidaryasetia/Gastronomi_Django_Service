from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from .controllers.file_upload_view import file_upload
from .controllers.file_upload_view import upload_architecture
from .controllers.file_upload_view import upload_model
from .controllers.file_download import file_download
from .controllers import food

urlpatterns = [
    path('hello', food.hello),
    path('predict', food.predict),
    path('predict-data', food.predict_with_data),
    path('multi-predict', food.multi_predict),
    path('file-upload', file_upload , name='file'),
    path('upload-architecture', upload_architecture ),
    path('upload-model', upload_model ),
    path('file-download/<str:file_name>/', file_download, name='file_download'),
]

# urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
