from django.urls import path
from . import views

urlpatterns = [
    path('register/', views.register_user, name='register_user'),
    path('register-face/', views.register_user, name='register_face'),  

    path('add-organization/', views.add_organization, name='add_organization'),

    path('wave-mark/', views.wave_mark_attendance, name='wave_mark_attendance'),
    path("mark/", views.mark_attendance, name="mark_attendance"),
    path("touchless-mark/", views.touchless_mark_attendance, name="touchless_mark_attendance"),
    path("mark-face/", views.mark_attendance, name="mark_face"),
    
    # path('mark/', views.mark_attendance, name='mark_attendance'),
    # path('touchless_mark/', views.touchless_mark_attendance, name='touchless_mark_attendance'),
    # path('mark-face/', views.mark_attendance, name='mark_face'),
    # path("touchless-mark/", views.touchless_mark_attendance, name="touchless_mark_attendance"), 

    path('find-twin/', views.find_twin, name='find_my_twin'),

]
