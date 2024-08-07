import sqlite3
import io
import os
import shutil
from django.core.exceptions import ValidationError
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, authenticate, logout
from django.http import HttpResponseForbidden
from django.conf import settings
from keras.models import load_model
from contextlib import redirect_stdout
from .forms import UploadFileForm, CustomUserCreationForm, ProcessDataForm, BuildModelForm, BuildSequentialForm, TrainModelForm
from .models import create_custom_db, Metadata
from .tasks import create_custom_dataset, create_model_instances, train_model
from .site_functions import get_latest_commit_info, upload_file, get_common_columns
from .db_functions import get_db_file_path, fetch_sample_dataset, save_metadata
from .model_functions import build_sequential_model, save_model, load_training_history, plot_metrics
from .form_functions import fetch_process_data_form_choices, process_sequential_model_form, fetch_train_model_form_choices

# View for the users dashboard
def home(request):
    message = get_latest_commit_info()
    context = {
        'message': message
    }    
    return render(request, 'home.html', context)

# View for the signup page
def signup(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            raw_password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=raw_password)
            login(request, user)
            return redirect('home')
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

# View that handles the process_data form
# Functions: populate_process_data_form(), get_common_columns(), fetch_process_data_form_choices(), create_custom_dataset(), create_custom_db(), create_model_instances(), get_db_file_path(), fetch_sample_dataset)_, populate_process_data_form()
def process_data_form(request):
    # Populates the form
    if request.method == 'POST':
        feature_count = int(request.POST.get('features', 1))
        selected_files = request.POST.getlist('files')
        common_columns = get_common_columns(selected_files)
        form = ProcessDataForm(request.POST, feature_count=feature_count, common_columns=common_columns)

        if form.is_valid():
            # Fetches form variables
            file_paths, dataset_type, start_row, end_row, features, feature_eng_choices, title, comment = fetch_process_data_form_choices(form)

            # Attempts to create the dataset, create a db, and save dataset to db
            dataset = create_custom_dataset(file_paths, features, start_row, end_row, feature_eng_choices)
            db = create_custom_db(title, dataset)
            dataset_saved = create_model_instances(dataset, db) # Converts the dataframe to model instances, then commits to db

            # Handles dataset saved successfully
            if dataset_saved:
                user = request.user
                file_path = get_db_file_path()
                save_metadata(title, comment, user, file_path, dataset_type)
                data, features = fetch_sample_dataset(title, 50)
                return render(request, 'datasets/sample_dataset.html', {'title': title, 'data': data, 'features': features})
            else:
                print(f'dataset not saved:')
        else:
            for field, errors in form.errors.items():
                for error in errors:
                    print(f"Error in {field}: {error}") 
    else:
        feature_count = 1
        common_columns = get_common_columns([])
        form = ProcessDataForm(feature_count=feature_count, common_columns=common_columns)

    feature_count_range = range(feature_count)
    context = {
        'form': form,
        'feature_count': feature_count_range
    }
    return render(request, 'datasets/process_data_form.html', context)

# View that handles the logic for the build model form
# Functions: fetch_build_model_form_choices(), build_sequential_model(), build_xgboost_model(), build_mamba_model()
def build_model_form(request):
    if request.method == 'POST':
        form = BuildModelForm(request.POST)
    else:
        form = BuildModelForm()
    return render(request, 'models/build_model_form.html', {'form': form})

# Function that handles the build model form
def handle_build_model_form(request):
    if request.method == 'POST':
        form= BuildModelForm(request.POST)
        if form.is_valid():
            model_title = form.cleaned_data['model_title']
            comment = form.cleaned_data['comment']
            model_type = form.cleaned_data['model_type']
            dataset_id = form.cleaned_data['feature_dataset']
            
            if model_type == 'sequential':
                seq_form = BuildSequentialForm(request.POST)
                if seq_form.is_valid():
                    input_shape, nodes, layer_types, activations, optimizer, loss, metrics = process_sequential_model_form(seq_form, dataset_id)
                    user = request.user
                    model = build_sequential_model(model_title, user, comment, layer_types, input_shape, nodes, activations, optimizer, loss, metrics)
                    if model:
                        model_metadata = Metadata.objects.get(title=model_title)
                        return redirect(view_model(request, model_id=model_metadata.id))
                else:
                    print(f'Form not valid: {seq_form.errors}')
        else:
            # Handle form errors
            print(f'Form not valid: {form.errors}')
            return render(request, 'models/build_model_form.html', {'form': form})
    else:
        form = BuildModelForm()
    return render(request, 'models/build_model_form.html', {'form': form})

# View that handles the train_model form
# Functions: fetch_train_model_form_choices(), populate_train_model_form()
def train_model_form(request):
    if request.method == 'POST':
        form = TrainModelForm(request.POST)
        if form.is_valid():
            title, comment, features, outputs, model, batch_size, epochs, verbose, validation_split = fetch_train_model_form_choices(form)
            history, model = train_model(features, outputs, model, batch_size, epochs, verbose, validation_split)
            user = request.user
            save_model(title, model, history, 'trained', user, comment)
            fig_url = plot_metrics(title, history.history)

            return render(request, 'models/evaluate_model.html', {'fig_url': fig_url})
    else:
        form = TrainModelForm()

    return render(request, 'models/train_model_form.html', {'form': form})

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
            #conn = sqlite3.connect(dataset_metadata.file_path)
            conn = sqlite3.connect(get_db_file_path())
            cursor = conn.cursor()
            # Drop the table
            table_name = f'myapp_{dataset_metadata.title}'
            cursor.execute(f'DROP TABLE IF EXISTS {table_name}"')
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
    
    # Extract layer information
    model_layers = []
    for layer in model.layers:
        layer_info = {
            'name': layer.name,
            'input_shape': layer.input_shape if hasattr(layer, 'input_shape') else None,
            'output_shape': layer.output_shape if hasattr(layer, 'output_shape') else None,
            'num_params': layer.count_params()
        }
        model_layers.append(layer_info)
    
    context = {
        'model_metadata': model_metadata,
        'model_summary': model_summary,
        'model_layers': model_layers,
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


