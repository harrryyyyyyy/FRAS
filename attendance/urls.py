from django.urls import path
from . import views

urlpatterns = [
    path('register/', views.register_user, name='register_user'),
    path('register-face/', views.register_user, name='register_face'),  

    path('add-organization/', views.add_organization, name='add_organization'),
    
    path('mark/', views.mark_attendance, name='mark_attendance'),
    path('mark-face/', views.mark_attendance, name='mark_face'),
]
