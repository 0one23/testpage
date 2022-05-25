from django.urls import path
from . import views

urlpatterns = [
    path('', views.title, name='title'),
    path('home', views.home, name='home'),
    path('home', views.add_questions, name='add_questions'),
    path('select_stage', views.select_stage, name='select_stage'),
    path('battle_disease', views.battle_disease, name='battle_disease'),
]