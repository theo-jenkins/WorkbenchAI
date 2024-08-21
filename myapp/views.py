import sqlite3
import io
import os
import shutil
from django.core.exceptions import ValidationError
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login as auth_login
from django.http import HttpResponseForbidden
from django.conf import settings
from keras.models import load_model
from contextlib import redirect_stdout
from .forms import UploadFileForm, CustomAuthenticationForm, CustomUserCreationForm, ProcessDataForm, BuildModelForm, TrainModelForm
from .models import CustomUser, Metadata
from .site_functions import get_latest_commit_info, upload_file
from .db_functions import fetch_sample_dataset
from .model_functions import load_training_history, plot_metrics

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
                error_message = "Invalid username or password, user is none"
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
# Functions: upload_file()
def upload_data_form(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES) #Creates the upload file form instance
        files = request.FILES.getlist('file_field')
        if form.is_valid():
            for file in files:
                try:
                    upload_file(file)
                except ValidationError as e:
                    form.add_error('file_field', e)
                    return render(request, 'datasets/upload_data_form.html', {'form': form})
            return redirect('home')
    else:
        form = UploadFileForm()
    return render(request, 'datasets/upload_data_form.html', {'form': form})

# View that renders the initial process data form
def process_data_form(request):
    if request.method == 'POST':
        form = ProcessDataForm(request.POST)
    else:
        form = ProcessDataForm()
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
    if request.method == 'POST':
        form = TrainModelForm(request.POST)
    else:
        form = TrainModelForm()
    return render(request, 'models/train_models/train_model_form.html', {'form': form})

# View the feature and output datasets created by the signed in user
def view_datasets(request):
    user = request.user
    feature_datasets = Metadata.objects.filter(user=user, tag='features')
    output_datasets = Metadata.objects.filter(user=user, tag='outputs')
    user_datasets = feature_datasets | output_datasets
    context = {
        'datasets': user_datasets
    }

    return render(request, 'datasets/view_datasets.html', context)

# View an individual dataset
def view_dataset(request, dataset_id):
    dataset_metadata = get_object_or_404(Metadata, id=dataset_id)
    if dataset_metadata.user != request.user:
        return HttpResponseForbidden
    
    sample_size = 50
    title = dataset_metadata.title
    data, features = fetch_sample_dataset(title, sample_size)
    return render(request, 'datasets/sample_dataset.html', {'title': title, 'data': data, 'features': features})

# View to handle the delete dataset logic
def delete_dataset(request, dataset_id):
    dataset_metadata = get_object_or_404(Metadata, id=dataset_id)
    if dataset_metadata.user != request.user:
        return HttpResponseForbidden()
    if request.method == 'POST':
        try:
            # Connect to the SQLite database
            conn = sqlite3.connect(dataset_metadata.file_path)
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
    user = request.user
    untrained_models = Metadata.objects.filter(user=user, tag='untrained')
    trained_models = Metadata.objects.filter(user=user, tag='trained')
    user_models = untrained_models | trained_models
    context = {
        'models': user_models
    }
    return render(request, 'models/view_models.html', context)

# View the metrics of a model
def view_model(request, model_id):
    model_metadata = get_object_or_404(Metadata, id=model_id)
    if model_metadata.user != request.user:
        return HttpResponseForbidden
    model = load_model(model_metadata.file_path)
    
    # Capture the model summary as a string
    with io.StringIO() as buf, redirect_stdout(buf):
        model.summary()
        model_summary = buf.getvalue()

    # Extract optimizer, loss function, and metrics
    optimizer = model.optimizer.__class__.__name__ if model.optimizer else "No optimizer found"
    loss = model.loss if model.loss else "No loss function found"
    metrics = [metric.name if hasattr(metric, 'name') else str(metric) for metric in model.metrics] if model.metrics else ["No metrics found"]
    
    context = {
        'model_metadata': model_metadata,
        'model_summary': model_summary,
        'optimizer': optimizer,
        'loss': loss,
        'metrics': metrics,
    }
    
    return render(request, 'models/view_model_summary.html', context)

# View to handle the delete model logic
def delete_model(request, model_id):
    model_metadata = get_object_or_404(Metadata, id=model_id)
    if model_metadata.user != request.user:
        return HttpResponseForbidden
    
    tag = model_metadata.tag
    title = model_metadata.title
    model_file_path = os.path.join(settings.MODEL_ROOT, tag, title)
    figures_dir_path = os.path.join(settings.FIGURES_ROOT, title)
    
    if request.method == 'POST':
        try:
            # Remove model file if it exists
            if os.path.exists(model_file_path):
                shutil.rmtree(model_file_path)
            else:
                print(f'Model file {model_file_path} does not exist.')
            
            # Remove figures directory if it exists
            if os.path.exists(figures_dir_path):
                shutil.rmtree(figures_dir_path)
            else:
                print(f'Figures directory {figures_dir_path} does not exist.')
            
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
    model_metadata = get_object_or_404(Metadata, id=model_id)
    if model_metadata.user != request.user:
        return HttpResponseForbidden
    
    # Fetches the models history as a plot
    model_title = model_metadata.title
    history = load_training_history(model_title)
    fig_url = plot_metrics(model_title, history)

    return render(request, 'models/evaluate_model.html', {'fig_url': fig_url})


