# bot_detection/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('upload-data/', views.upload_data, name='upload_data'),
    path('capture_data/',views.capture_data_view,name='capture_data'),
    
]