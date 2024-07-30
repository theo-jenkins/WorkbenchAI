from django.urls import path
from django.contrib.auth import views as auth_views
from . import views
from .form_functions import update_process_data_form, update_build_model_form

urlpatterns = [
    path('', views.home, name='home'),
    path('login/', auth_views.LoginView.as_view(), name='login'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
    path('signup/', views.signup, name='signup'),
    path('upload_data_form/', views.upload_data_form, name='upload_data_form'),
    path('process_data_form/', views.process_data_form, name='process_data_form'),
    path('update_process_data_form/', update_process_data_form, name='update_process_data_form'),
    path('build_model_form/', views.build_model_form, name='build_model_form'),
    path('update_build_model_form/', update_build_model_form, name='update_build_model_form'),
    path('train_model_form/', views.train_model_form, name='train_model_form'),
    path('view_datasets/', views.view_datasets, name='view_datasets'),
    path('view_models/', views.view_models, name='view_models'),
    path('view_model/<int:model_id>/', views.view_model, name='view_model'),
    path('view_dataset/<int:dataset_id>/', views.view_dataset, name='view_dataset'),
    path('delete_dataset/<int:dataset_id>/', views.delete_dataset, name='delete_dataset'),
    path('delete_model/<int:model_id>/', views.delete_model, name='delete_model'),
]