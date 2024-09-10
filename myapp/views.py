import sqlite3
import io
import os
import shutil
from django.core.exceptions import ValidationError
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login as auth_login
from django.http import HttpResponseForbidden,HttpResponseNotFound
from django.conf import settings
from contextlib import redirect_stdout
from .forms import UploadFileForm, CustomAuthenticationForm, CustomUserCreationForm, ProcessDataForm, BuildModelForm, TrainModelForm, MakePredictionForm
from .models import CustomUser, FileMetadata, DatasetMetadata, ModelMetadata
from .site_functions import get_latest_commit_info, upload_file, get_file_choices, file_exists
from .db_functions import fetch_sample_dataset, save_file_metadata, get_db_file_path
from .model_functions import load_training_history, plot_metrics, fetch_gpu_info, prepare_model

# View for the users dashboard
def home(request):
    message = get_latest_commit_info()
    context = {
        'message': message
    }    
    return render(request, 'home.html', context)

# View for the login page
def login(request):
    # Check if any accounts exist
    if CustomUser.objects.count() == 0:
        return redirect('signup')  # Redirect to signup if no users exist

    if request.method == 'POST':
        form = CustomAuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            if user is not None:
                auth_login(request, form.get_user())
                return redirect('home')
            else:
                # Add an error message if authentication fails
                error_message = 'Invalid username or password, user is none'
                return render(request, 'authentication/login.html', {'form': form, 'error_message': error_message})
        else:
            # Add an error message if authentication fails
            error_message = "Invalid username or password, form not valid"
            return render(request, 'authentication/login.html', {'form': form, 'error_message': error_message})
    else:
        form = CustomAuthenticationForm()
        return render(request, 'authentication/login.html', {'form': form})

# View for the signup page
def signup(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            email = form.cleaned_data.get('email')
            raw_password = form.cleaned_data.get('password1')
            user = authenticate(username=email, password=raw_password)
            if user is not None:
                auth_login(request, user)  # Only log in if the user is authenticated successfully
                return redirect('home')
            else:
                form.add_error(None, "Authentication failed. Please check your credentials.")
    else:
        form = CustomUserCreationForm()
    return render(request, 'authentication/signup.html', {'form': form})

# View that handles the upload_data logic
def upload_files_form(request):
    user = request.user
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES) # Creates the upload file form instance
        files = request.FILES.getlist('file_field')
        if form.is_valid():
            for uploaded_file in files:
                try:
                    # Validate and save the file if it's unique
                    if not file_exists(uploaded_file):
                        file = upload_file(uploaded_file)
                        if file:
                            file_path = os.path.join(settings.USER_ROOT, file.name)
                            save_file_metadata(user, file.name, file_path, 'csv')
                    else:
                        print(f"File {uploaded_file.name} already exists. Skipping upload.")
                except ValidationError as e:
                    return render(request, 'uploads/upload_files_form.html', {'form': form})
            return redirect('upload_files_form')
    else:
        form = UploadFileForm()

    user_files = get_file_choices(user)
    return render(request, 'uploads/upload_files_form.html', {'form': form, 'files': user_files})

# Function that deletes a user uploaded file
def delete_file(request, file_id):
    file_metadata = get_object_or_404(FileMetadata, id=file_id)
    
    # Check if the user is the owner of the file
    if file_metadata.user != request.user:
        return HttpResponseForbidden("You are not allowed to delete this file.")
    
    file_path = os.path.join(settings.MEDIA_ROOT, file_metadata.file_path)
    if request.method == 'POST':
        try:    
            # Check if the file exists before attempting to delete it
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    return HttpResponseNotFound(f"Error occurred while trying to delete the file: {e}")
            else:
                return HttpResponseNotFound("File not found.")
            
            # Delete the file metadata from the database
            file_metadata.delete()
            return redirect('upload_files_form')
        except Exception as e:
            return HttpResponseNotFound(f"Error occurred while trying to delete the file: {e}")
        
    # Redirect back to the upload form page after deletion
    return render(request, 'uploads/confirm_file_delete.html', {'file': file_metadata})

# View that renders the initial process data form
def process_data_form(request):
    user = request.user
    if request.method == 'POST':
        form = ProcessDataForm(request.POST, user=user)
    else:
        form = ProcessDataForm(user=user)
    return render(request, 'datasets/process_data_form.html', {'form': form})

# View that renders the initial build model form
def build_model_form(request):
    if request.method == 'POST':
        form = BuildModelForm(request.POST)
    else:
        form = BuildModelForm()
    return render(request, 'models/build_models/build_model_form.html', {'form': form})

# View that renders the initial train model form
def train_model_form(request):
    message = fetch_gpu_info()
    if request.method == 'POST':
        form = TrainModelForm(request.POST)
    else:
        form = TrainModelForm()
    return render(request, 'models/train_models/train_model_form.html', {'form': form, 'message': message})

# View the feature and output datasets created by the signed in user
def view_datasets(request):
    user = request.user
    feature_datasets = DatasetMetadata.objects.filter(user=user, tag='features')
    output_datasets = DatasetMetadata.objects.filter(user=user, tag='outputs')
    user_datasets = feature_datasets | output_datasets
    context = {
        'datasets': user_datasets
    }

    return render(request, 'datasets/view_datasets.html', context)

# View an individual dataset
def view_dataset(request, dataset_id):
    dataset_metadata = get_object_or_404(DatasetMetadata, id=dataset_id)
    if dataset_metadata.user != request.user:
        return HttpResponseForbidden
    
    sample_size = 50
    title = dataset_metadata.title
    data, features = fetch_sample_dataset(title, sample_size)
    return render(request, 'datasets/sample_dataset.html', {'title': title, 'data': data, 'features': features})

# View to handle the delete dataset logic
def delete_dataset(request, dataset_id):
    dataset_metadata = get_object_or_404(DatasetMetadata, id=dataset_id)
    if dataset_metadata.user != request.user:
        return HttpResponseForbidden()
    if request.method == 'POST':
        try:
            # Connect to the SQLite database
            db_file_path = get_db_file_path()
            conn = sqlite3.connect(db_file_path)
            cursor = conn.cursor()
            # Drop the table
            table_name = f'myapp_{dataset_metadata.title}'
            cursor.execute(f'DROP TABLE IF EXISTS "{table_name}"')
            conn.commit()
            conn.close()
            
            # Delete the metadata object
            dataset_metadata.delete()
            return redirect('view_datasets')
        except Exception as e:
            print(f"Error deleting table: {e}")
            return HttpResponseForbidden("An error occurred while deleting the dataset.")

    return render(request, 'datasets/confirm_dataset_delete.html', {'dataset': dataset_metadata})

# View the trained and untrained models created by the signed in user
def view_models(request):
    # View all user created models sorted by title
    user = request.user
    models = ModelMetadata.objects.filter(user=user).order_by('title')

    context = {
        'models': models
    }
    return render(request, 'models/view_models.html', context)

# View the metrics of a model
def view_model(request, model_id):
    model_metadata = get_object_or_404(ModelMetadata, id=model_id)
    if model_metadata.user != request.user:
        return HttpResponseForbidden
    
    model = prepare_model(model_metadata.id) 
    
    # Capture the model summary as a string
    with io.StringIO() as buf, redirect_stdout(buf):
        model.summary()
        model_summary = buf.getvalue()

    # Extract optimizer, loss function, and metrics
    optimizer = model.optimizer.__class__.__name__ if model.optimizer else 'No optimizer found'
    loss = model.loss if model.loss else 'No loss function found'
    
    context = {
        'model_metadata': model_metadata,
        'model_summary': model_summary,
        'optimizer': optimizer,
        'loss': loss,
    }
    
    return render(request, 'models/view_model_summary.html', context)

# View to handle the delete model logic
def delete_model(request, model_id):
    model_metadata = get_object_or_404(ModelMetadata, id=model_id)
    if model_metadata.user != request.user:
        return HttpResponseForbidden
    
    title = model_metadata.title
    model_version_dir = os.path.join(settings.MODEL_ROOT, title, f'Version_{model_metadata.version}')
    figures_dir_path = os.path.join(settings.FIGURES_ROOT, title)
    model_dir_path = os.path.join(settings.MODEL_ROOT, title)
    
    if request.method == 'POST':
        try:
            # Remove model version directory if it exists
            if os.path.exists(model_version_dir):
                shutil.rmtree(model_version_dir)
                print(f'Removed model version directory: {model_version_dir}')
            else:
                print(f'Model version directory {model_version_dir} does not exist.')
            
            # Remove figures directory if it exists
            if os.path.exists(figures_dir_path):
                shutil.rmtree(figures_dir_path)
                print(f'Removed figures directory: {figures_dir_path}')
            else:
                print(f'Figures directory {figures_dir_path} does not exist.')
            
            # Check if any other versions of the model still exist
            remaining_versions = os.listdir(model_dir_path)
            if not remaining_versions:
                # If no other versions exist, remove the entire model directory
                shutil.rmtree(model_dir_path)
                print(f'Removed model directory: {model_dir_path}')
            else:
                print(f'Remaining versions found: {remaining_versions}')
            
            # Delete model metadata from the database
            model_metadata.delete()

            return redirect('view_models')
        except Exception as e:
            print(f'Error deleting model: {e}')
            return HttpResponseForbidden('An error occured while deleting the model.')
    return render(request, 'models/confirm_model_delete.html', {'model': model_metadata})

# View the models training history, loss and accuracy
def evaluate_model(request, model_id):
    # Fetches the model metadata object
    model_metadata = get_object_or_404(ModelMetadata, id=model_id)
    if model_metadata.user != request.user:
        return HttpResponseForbidden
    
    # Fetches the models history as a plot
    model_title = model_metadata.title
    model_version = model_metadata.version
    history = load_training_history(model_title, model_version)
    fig_url = plot_metrics(model_title, model_version, history)

    return render(request, 'models/evaluate_model.html', {'fig_url': fig_url})

# View the make prediction screen
def make_prediction_view(request):
    user = request.user

    if request.method == 'POST':
        form = MakePredictionForm(request.POST, user=user)
    else:
        form = MakePredictionForm(user=user)
    
    return render(request, 'models/make_prediction_form.html', {'form': form})
