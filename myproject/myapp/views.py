import sqlite3
import io
import os
from django.core.exceptions import ValidationError
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, authenticate, logout
from django.http import HttpResponseForbidden
from django.conf import settings
from keras.models import load_model
from contextlib import redirect_stdout
from .forms import UploadFileForm, CustomUserCreationForm, ProcessDataForm, TrainModelForm, BuildModelForm
from .models import create_custom_db, Metadata
from .tasks import create_custom_dataset, create_model_instances, train_model
from .site_functions import upload_file, get_common_columns
from .db_functions import get_db_file_path, fetch_sample_dataset, save_metadata
from .model_functions import build_model, save_model, load_training_history, plot_metrics
from .form_functions import fetch_process_data_form_choices, process_build_model_form, fetch_train_model_form_choices, populate_train_model_form

# View for the users dashboard
def home(request):
    return render(request, 'home.html')

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
    return render(request, 'registration/signup.html', {'form': form})

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
                    return render(request, 'upload_data_form.html', {'form': form})
            return redirect('home')
    else:
        form = UploadFileForm()
    return render(request, 'upload_data_form.html', {'form': form})

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
            dataset_saved = create_model_instances(dataset, db)

            # Handles dataset saved successfully
            if dataset_saved:
                user = request.user
                file_path = get_db_file_path()
                metadata = save_metadata(title, comment, user, file_path, dataset_type)
                db_data, features = fetch_sample_dataset(db, 50)
                return render(request, 'sample_dataset.html', {'title': title, 'db_data': db_data, 'features': features})
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
    return render(request, 'process_data_form.html', context)

# View that handles the logic for the build_model form
# Functions: fetch_build_model_form_choices(), build_model()
def build_model_form(request):
    if request.method == 'POST':
        hidden_layer_count=int(request.POST.get('hidden_layers', 1))
        form = BuildModelForm(request.POST, hidden_layer_count=hidden_layer_count)

        if form.is_valid():
            title, comment, nodes, layer_types, activations, optimizer, loss, metrics = process_build_model_form(form)
            user = request.user
            model = build_model(title, user, comment, layer_types, nodes, activations, optimizer, loss, metrics)
            return redirect('home')
        else:
            print(f'form errors: {form.errors}')
            
    else:
        hidden_layer_count = 1
        form = BuildModelForm(hidden_layer_count=hidden_layer_count)

    layer_count_range = range(hidden_layer_count)
    return render(request, 'build_model_form.html', {'form': form, 'layer_count_range': layer_count_range})


# View that handles the train_model form
# Functions: fetch_train_model_form_choices(), populate_train_model_form()
def train_model_form(request):
    if request.method == 'POST':
        form = TrainModelForm(request.POST)
        populate_train_model_form(form)
        if form.is_valid():
            title, comment, features, outputs, model, batch_size, epochs, verbose, validation_split = fetch_train_model_form_choices(form)
            history, model = train_model(features, outputs, model, batch_size, epochs, verbose, validation_split)
            user = request.user
            save_model(title, model, history, 'trained', user, comment)

            return redirect('home')
    else:
        form = TrainModelForm()
        populate_train_model_form(form)

    return render(request, 'train_model_form.html', {'form': form})

# View the feature and output datasets created by the signed in user
def view_datasets(request):
    user = request.user
    feature_datasets = Metadata.objects.filter(user=user, tag='features')
    output_datasets = Metadata.objects.filter(user=user, tag='outputs')
    user_datasets = feature_datasets | output_datasets
    context = {
        'datasets': user_datasets
    }

    return render(request, 'view_datasets.html', context)

# View an individual dataset
def view_dataset(request, dataset_id):
    dataset = get_object_or_404(Metadata, id=dataset_id)
    if dataset.user != request.user:
        return HttpResponseForbidden
    data, columns = fetch_sample_dataset(dataset, 50)
    return render('sample_dataset.html', {'data': data, 'columns': columns})

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
            cursor.execute(f'DROP TABLE IF EXISTS "myapp_{dataset_metadata.title}"')
            conn.commit()
            conn.close()
            
            # Delete the metadata object
            dataset_metadata.delete()
            return redirect('view_datasets')
        except Exception as e:
            print(f"Error deleting table: {e}")
            return HttpResponseForbidden("An error occurred while deleting the dataset.")

    return render(request, 'confirm_dataset_delete.html', {'dataset': dataset_metadata})

# View the trained and untrained models created by the signed in user
def view_models(request):
    user = request.user
    untrained_models = Metadata.objects.filter(user=user, tag='untrained')
    trained_models = Metadata.objects.filter(user=user, tag='trained')
    user_models = untrained_models | trained_models
    context = {
        'models': user_models
    }
    return render(request, 'view_models.html', context)

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
    
    return render(request, 'view_model_summary.html', context)

# View to handle the delete model logic
def delete_model(request, model_id):
    model_metadata = get_object_or_404(Metadata, id=model_id)
    if model_metadata.user != request.user:
        return HttpResponseForbidden
    if request.method == 'POST':
        try:
            if os.path.exists(model_metadata.file_path):
                os.remove(model_metadata.file_path)
            else:
                print(f'File {model_metadata.file_path} does not exist.')
            model_metadata.delete()
            return redirect('view_models')
        except Exception as e:
            print(f'Error deleting model: {e}')
            return HttpResponseForbidden('An error occured while deleting the model.')
    return render(request, 'confirm_model_delete.html', {'model': model_metadata})

# View the models training history, loss and accuracy
def evaluate_model(request, model_id):
    model_metadata = get_object_or_404(Metadata, id=model_id)
    if model_metadata.user != request.user:
        return HttpResponseForbidden
    
    # Path to the figure file
    fig_dir = os.path.join(settings.FIGURES_ROOT, model_metadata.title)
    fig_path = os.path.join(fig_dir, 'metrics.png')

    # Check if the figure already exists
    if not os.path.exists(fig_path):
        os.makedirs(fig_dir)
        history = load_training_history(model_metadata.title)
        fig_path = plot_metrics(history, fig_path)
        print(f'Figure created at: {fig_path}')
    else:
        print(f'Figure already exists at: {fig_path}')

    # Creates the url for the figure
    fig_url = os.path.join(settings.FIGURES_URL, model_metadata.title, 'metrics.png')

    return render(request, 'evaluate_model.html', {'fig_url': fig_url})


