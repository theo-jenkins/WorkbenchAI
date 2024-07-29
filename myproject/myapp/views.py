import pandas as pd
import tensorflow as tf
import os
import zipfile
import requests
import re
from django.core.exceptions import ValidationError
from django.conf import settings
from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from django.http import JsonResponse
from django.db import DatabaseError, transaction, models
from django.utils import timezone
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from alpha_vantage.timeseries import TimeSeries
from .forms import UploadFileForm, CustomUserCreationForm, ProcessDataForm, BuildModelForm, TrainModelForm, ProcessTickerForm
from .models import create_custom_db, Metadata
from .tasks import upload_files, create_custom_dataset, create_model_instances, train_model

# View that allows the user to access there dashboard
def home(request):
    return render(request, 'home.html')

# View that allows the user to create an account
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

# Form that allows user to upload multiple files
def upload_data_form(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES) #Creates the upload file form instance
        files = request.FILES.getlist('file_field')
        if form.is_valid():
            result = upload_files.delay(files)
            return render('home.html', {'result': result})
    else:
        form = UploadFileForm()
    return render(request, 'upload_data_form.html', {'form': form})

# Function that returns the file names in the /uploaded_files folder
def get_uploaded_files():
    upload_dir = os.path.join(settings.MEDIA_ROOT)
    if not os.path.exists(upload_dir):
        return []
    
    file_list = [
        (file, os.path.relpath(os.path.join(root, file), upload_dir))
        for root, _, files in os.walk(upload_dir)
        for file in files
    ]
    return file_list

# Function that finds the common columns between .csv files selected
def get_common_columns(file_paths):
    upload_dir = settings.MEDIA_ROOT
    common_columns = None
    print(f'file_paths: {file_paths}')
    for file_path in file_paths:
        print(f'file_path: {file_path}')
        full_path = os.path.join(upload_dir, file_path)
        # Read only the first row to get the columns
        df = pd.read_csv(full_path, nrows=0)  # nrows=0 loads no data rows, only headers

        if common_columns is None:
            common_columns = set(df.columns)
        else:
            common_columns.intersection_update(df.columns)

        if not common_columns:
            break  # No need to proceed if there are no common columns left

    return sorted(common_columns) if common_columns else []

# Function finds the maximum rows in the .csv files selected
def get_max_rows(file_paths):
    upload_dir = settings.MEDIA_ROOT
    full_paths = []

    # Traverse the directory to find all files
    for root, _, files in os.walk(upload_dir):
        for file in files:
            if file in file_paths:
                full_path = os.path.join(root, file)
                full_paths.append(full_path)
    
    # Reads the .csv files and finds the maximum length
    max_rows = 0
    for full_path in full_paths:
        df = pd.read_csv(full_path)
        max_rows += len(df)

    return max_rows

# Function that checks what files are selected and updates the form dynamically
def update_process_data_form(request):
    if request.method == 'POST':
        selected_files = request.POST.getlist('files[]')
        print(f'selected_files: {selected_files}')
        try:
            max_rows = get_max_rows(selected_files)
            common_columns = get_common_columns(selected_files)
            return JsonResponse({'columns': common_columns, 'max_rows': max_rows})
        except Exception as e:
            print(f'Error processing files: {e}')
            return JsonResponse({'columns': [], 'max_rows': 0})

    return JsonResponse({'columns': [], 'max_rows': 0})

# Function that fetches the file path of the Django db
def get_db_file_path():
    db_path = settings.DATABASES['default']['NAME']
    return db_path

# Function that fetches the variables from the completed process_data form
def fetch_process_data_form_choices(form):
    files = form.cleaned_data['files']
    columns = form.cleaned_data['columns']
    dataset_type = form.cleaned_data['dataset_type']
    start_row = form.cleaned_data['start_row']
    end_row = form.cleaned_data['end_row']
    feature_eng = form.cleaned_data['feature_eng']
    title = form.cleaned_data['db_title']
    comment = form.cleaned_data['comment']

    return files, columns, dataset_type, start_row, end_row, feature_eng, title, comment

# Function that fetches a sample of a specified database
def fetch_sample_dataset(db, sample_size):
    db_data = db.objects.all().values()[:sample_size]  # Get the first 50 rows
    columns = db_data[0].keys() if db_data else []  # Get column names

    return db_data, columns

# Function that fills in the choices for the process data form
def populate_process_data_form_choices(form):
    file_list = get_uploaded_files()
    form.fields['files'].choices = file_list
    form.fields['feature_eng'].choices = [
        ('handle_missing', 'Handle missing values'), 
        ('normalize', 'Normalization'), 
        ('standardize', 'Standardization')
    ]
    form.fields['dataset_type'].choices = [
        ('features', 'Features (Input Data)'),
        ('outputs', 'Targets (Output Data)')
    ]
    return form

# View that handles the create_custom_dataset logic
def process_data_form(request):
    # Populates the form
    if request.method == 'POST':
        form = ProcessDataForm(request.POST)
        form = populate_process_data_form_choices(form)

        # Get the selected files
        selected_files = request.POST.getlist('files')
        if selected_files:
            common_columns = get_common_columns(selected_files)
            form.fields['columns'].choices = [(col, col) for col in common_columns]

        if form.is_valid():
            # Fetches form variables
            files, columns, dataset_type, start_row, end_row, feature_eng, title, comment = fetch_process_data_form_choices(form)

            # Attempts to create the dataset, create a db, and save dataset to db
            dataset = create_custom_dataset.delay(files, columns, start_row, end_row, feature_eng)
            db = create_custom_db.delay(title, dataset)
            dataset_saved = create_model_instances.delay(dataset, db)

            # Handles dataset saved successfully
            if dataset_saved:
                user = request.user
                file_path = get_db_file_path()
                metadata = save_metadata(title, comment, user, file_path, dataset_type)
                db_data, columns = fetch_sample_dataset(db, 50)
                return render(request, 'sample_dataset.html', {'title': title, 'db_data': db_data, 'columns': columns})
            else:
                print(f'dataset_not saved:')
    else:
        form = ProcessDataForm()
        form = populate_process_data_form_choices(form)

    return render(request, 'process_data_form.html', {'form': form})

# Function that handles the dynamic update for the build model form
def update_build_model_form(request):
    hidden_layers = int(request.POST.get('hidden_layers', 0))
    hidden_layer_form = BuildModelForm()

    layer_html = ""
    for i in range(hidden_layers):
        if i < hidden_layers - 1:
            # Render Hidden layers
            layer_html += f'<div><label for="layer_type_{i}">Hidden Layer Type {i+1}</label>{hidden_layer_form["layer_type"]}'
            layer_html += f'<label for="nodes_{i}">Nodes {i+1}</label>{hidden_layer_form["nodes"]}'
            layer_html += f'<label for="activation_{i}">Activation {i+1}</label>{hidden_layer_form["activation"]}</div>'

    return JsonResponse({'layer_html': layer_html})

# View that handles the build_model_form logic
def build_model_form(request):
    if request.method == 'POST':
        form = BuildModelForm(request.POST)
        input_form = BuildModelForm(request.POST, prefix='input')
        hidden_layer_form = BuildModelForm(request.POST, prefix='hidden_layer')
        output_form = BuildModelForm(request.POST, prefix='output')
        
        if form.is_valid() and input_form.is_valid() and hidden_layer_form.is_valid() and output_form.is_valid():
            print('form verified')
            title = form.cleaned_data['model_title']
            comment = form.cleaned_data['comment']
            optimizer = form.cleaned_data['optimizer']
            loss = form.cleaned_data['loss']
            metrics = form.cleaned_data['metrics']

            # Saves the form variables as lists
            layer_count = int(request.POST.get('layer_count', 0))
            activations = []
            layer_types = []
            nodes = []

            layer_types.append(input_form.cleaned_data['layer_type'])
            activations.append(input_form.cleaned_data['activation'])
            nodes.append(input_form.cleaned_data['features'])
            for i in range(layer_count):
                activation = request.POST.get(f'activation_{i}')
                layer_type = request.POST.get(f'layer_type_{i}')
                node = request.POST.get(f'nodes_{i}')
                activations.append(activation)
                layer_types.append(layer_type)
                nodes.append(node)
            layer_types.append(output_form.cleaned_data['layer_type'])
            activations.append(output_form.cleaned_data['activation'])
            nodes.append(output_form.cleaned_data['outputs'])

            print(f'nodes: {nodes}, layer_types: {layer_types}: activation: {activations}')
            
            # Attempts to build the model
            user = request.user
            model = build_model.delay(title, user, comment, layer_types, activations, nodes, optimizer, loss, metrics)
            if model.successful():
                print(model.result)
            return redirect('home')
        else:
            print(f'form errors: {form.errors}')
            
    else:
        form = BuildModelForm()
        input_form = BuildModelForm(prefix='input')
        hidden_layer_form = BuildModelForm(prefix='hidden_layer')
        output_form = BuildModelForm(prefix='output')

    return render(request, 'build_model_form.html', {'form': form, 'input_form': input_form, 'hidden_layer_form': hidden_layer_form, 'output_form': output_form})

# Function that fetches the variables from the completed train_model form
def fetch_train_model_form_choices(form):
    features = form.cleaned_data['feature_dataset']
    output = form.cleaned_data['training_dataset']
    batch_size = form.cleaned_data['batch_size']
    epochs = form.cleaned_data['epochs']
    verbose = form.cleaned_data['verbose']
    validation_split = form.cleaned_data['validation_split']

    return features, output, batch_size, epochs, verbose, validation_split

# Function that populates the train_model form
def populate_train_model_form(form):
    # Query for datasets tagged as 'features' or 'outputs'
    feature_datasets = Metadata.objects.filter(tag='features')
    training_datasets = Metadata.objects.filter(tag='outputs')
    
    # Query for models tagged as 'untrained'
    untrained_models = Metadata.objects.filter(tag='untrained')
    
    # Prepare choices as tuples (id, title)
    feature_choices = [(dataset.id, dataset.title) for dataset in feature_datasets]
    training_choices = [(dataset.id, dataset.title) for dataset in training_datasets]
    model_choices = [(model.id, model.title) for model in untrained_models]

    form.fields['feature_dataset'].choices = feature_choices
    form.fields['training_dataset'].choices = training_choices
    form.fields['model'].choices = model_choices
    form.fields['verbose'].choices = [
        (0, '0'),
        (1, '1'),
        (2, '2')
    ]
    return form

# View that handles the train_model form logic
def train_model_form(request):
    if request.method == 'POST':
        form = TrainModelForm(request.POST)
        if form.is_valid():
            features, output, model, batch_size, epochs, verbose, validation_split = fetch_train_model_form_choices(form)
            result = train_model.delay(features, output, model, batch_size, epochs, verbose, validation_split)
            if result.successful():
                print(result.result)
            title = form.cleaned_data['title']
            comment = form.cleaned_data['comment']
            user = request.user
            save_model.delay(title, model, 'trained', comment, user)
            print('model trained succesfully')
            return redirect('home')
    else:
        form = TrainModelForm()
        populate_train_model_form(form)

    return render(request, 'train_model_form.html', {'form': form})

# Function that uses AlphaVantage to search for ticker data
def search_ticker(api_key, ticker):
    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'SYMBOL_SEARCH',
        'keywords': ticker,
        'apikey': api_key
    }
    response = requests.get(url, params=params)
    data = response.json()

    print(data)
    
    return data

# Function which handles searching for a ticker form
def process_ticker_form(request):
    if request.method == 'POST':
        form = ProcessTickerForm(request.POST)
        if form.is_valid():
            ticker = form.cleaned_data['ticker']
            api_key = '6TZ6QYSNUILJQQ5K'
            results = search_ticker(api_key, ticker)
            return render(request, 'process_ticker_form.html', {'results': results})
    else:
        form = ProcessTickerForm()
    return render(request, 'process_ticker_form.html', {'form': form})

# Function that builds a keras neural network
def build_model(title, user, comment, layer_types, nodes, activations, optimizer, loss, metrics):
    model = Sequential()

    for i in range(len(layer_types)):
        if i == 0:
            # Adding the input layer
            if layer_types[i] == 'dense':
                model.add(Dense(nodes[i], input_dim=nodes[i], activation=activations[i]))
        else:
            # Adding hidden and output layers
            if layer_types[i] == 'dense':
                model.add(Dense(nodes[i], activation=activations[i]))

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Saves the model as a file
    save_model(title, model, 'untrained', user, comment)
    return model

# Function that saves a keras model based on if its trained or untrained
def save_model(title, model, trained_status, comment, user):
    # Determine the directory based on the training status
    if trained_status == 'trained':
        save_dir = os.path.join(settings.BASE_DIR, 'nn_models', 'trained')
    elif trained_status == 'untrained':
        save_dir = os.path.join(settings.BASE_DIR, 'nn_models', 'untrained')
    else:
        raise ValueError(f'Training status not recognized: {trained_status}')

    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Construct the file path
    model_path = os.path.join(save_dir, f'{title}.keras')

    # Save the model
    model.save(model_path)
    print(f'Model saved successfully to: {model_path}')

    # Saves the metadata
    save_metadata(title, comment, user, model_path, trained_status)

# Function that saves the metadata of a dataset or model
def save_metadata(title, comment, user, file_path, tag):
    # Django automatically generates a unique ID
    metadata = Metadata.objects.create(
        title=title,
        comment=comment,
        user=user,
        file_path=file_path,
        created_at=timezone.now(),
        tag=tag,
    )
    return metadata