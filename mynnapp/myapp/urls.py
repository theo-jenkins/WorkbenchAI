from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('login/', auth_views.LoginView.as_view(), name='login'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
    path('signup/', views.signup, name='signup'),
    path('upload_data_form/', views.upload_data_form, name='upload_data_form'),
    path('process_data_form/', views.process_data_form, name='process_data_form'),
    path('update_process_data_form/', views.update_process_data_form, name='update_process_data_form'),
    path('process_success/', views.process_success, name='process_success'),
    path('build_model_form/', views.build_model_form, name='build_model_form'),
    path('train_model_form', views.train_model_form, name='train_model_form'),
]