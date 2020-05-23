from django.urls import path

from fruitDetect import settings
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('results/', views.results, name='results')
]