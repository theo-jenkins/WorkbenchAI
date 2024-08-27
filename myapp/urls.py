from django.urls import path
from . import views
from .form_functions import handle_process_data_form, update_process_data_form, handle_build_model_form, update_build_model_form, update_sequential_model_form, handle_train_model_form, update_train_model_form, handle_make_prediction_form, update_make_prediction_form

urlpatterns = [
    path('', views.home, name='home'),
    path('login/', views.login, name='login'),
    path('signup/', views.signup, name='signup'),
    path('upload_files_form/', views.upload_files_form, name='upload_files_form'),
    path('delete_file/<int:file_id>/', views.delete_file, name='delete_file'),
    path('process_data_form/', views.process_data_form, name='process_data_form'),
    path('handle_process_data_form/', handle_process_data_form, name='handle_process_data_form'),
    path('update_process_data_form/', update_process_data_form, name='update_process_data_form'),
    path('build_model_form/', views.build_model_form, name='build_model_form'),
    path('handle_build_model_form/', handle_build_model_form, name='handle_build_model_form'),
    path('update_sequential_model_form/', update_sequential_model_form, name='update_sequential_model_form'),
    path('update_build_model_form/', update_build_model_form, name='update_build_model_form'),
    path('train_model_form/', views.train_model_form, name='train_model_form'),
    path('handle_train_model_form/', handle_train_model_form, name='handle_train_model_form'),
    path('update_train_model_form/', update_train_model_form, name='update_train_model_form'),
    path('view_datasets/', views.view_datasets, name='view_datasets'),
    path('view_dataset/<int:dataset_id>/', views.view_dataset, name='view_dataset'),
    path('delete_dataset/<int:dataset_id>/', views.delete_dataset, name='delete_dataset'),
    path('view_models/', views.view_models, name='view_models'),
    path('view_model/<int:model_id>/', views.view_model, name='view_model'),
    path('evaluate_model/<int:model_id>/', views.evaluate_model, name='evaluate_model'),
    path('delete_model/<int:model_id>/', views.delete_model, name='delete_model'),
    path('make_prediction_form/', views.make_prediction_view, name='make_prediction_view'),
    path('handle_make_prediction_form/', handle_make_prediction_form, name='handle_make_prediction_form'),
    path('update_make_prediction_form/', update_make_prediction_form, name='update_make_prediction_form'),
]