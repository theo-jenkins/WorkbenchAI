from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('login/', auth_views.LoginView.as_view(), name='login'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
    path('signup/', views.signup, name='signup'),
    path('upload_data_form/', views.upload_data_form, name='upload_data_form'),
    path('fetch_common_columns/', views.fetch_common_columns, name='fetch_common_columns'),
    path('process_data_form/', views.process_data_form, name='process_data_form'),
    path('process_success/', views.process_success, name='process_success'),
]