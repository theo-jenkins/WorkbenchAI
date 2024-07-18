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
from django.contrib.auth.decorators import login_required
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

def home(request):
    return render(request, 'home.html')

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

# Function that validates and saves the file and deletes the .zip if applicable
def upload_file(file):
    valid_extensions = ['.zip', '.csv']    
    # Check if the file has a valid extension
    if not any(file.name.endswith(ext) for ext in valid_extensions):
        raise ValidationError('Invalid file extension')
    
    try:
        # Ensure the uploaded_files directory exists
        upload_dir = os.path.join(settings.MEDIA_ROOT)
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save the uploaded file to the directory
        file_path = os.path.join(upload_dir, file.name)
        with open(file_path, 'wb+') as destination:
            for chunk in file.chunks():
                destination.write(chunk)
        # Handles .zip files
        if file.name.endswith('.zip'):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                # Extract only .csv files
                for zip_info in zip_ref.infolist():
                    if zip_info.filename.endswith('.csv'):
                        zip_info.filename = os.path.basename(zip_info.filename)  # Removes any directory structure
                        zip_ref.extract(zip_info, upload_dir)
            os.remove(file_path)  # Remove the .zip file after extraction
        return file  # Return the file object for further use if needed
    except Exception as e:
        raise ValidationError("There was an error uploading the file. Please try again.")

# Form that allows user to upload multiple files
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
    full_paths = []

    # Traverse the directory to find all files
    for root, _, files in os.walk(upload_dir):
        for file in files:
            if file in file_paths:
                full_path = os.path.join(root, file)
                full_paths.append(full_path)

    # Reads the .csv files and finds the common column headers
    dfs = [pd.read_csv(full_path) for full_path in full_paths]
    common_columns = set(dfs[0].columns)
    for df in dfs[1:]:
        common_columns.intersection_update(df.columns)
    return sorted(list(common_columns))

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
        try:
            max_rows = get_max_rows(selected_files)
            common_columns = get_common_columns(selected_files)
            return JsonResponse({'columns': common_columns, 'max_rows': max_rows})
        except Exception as e:
            print(f'Error processing files: {e}')
            return JsonResponse({'columns': [], 'max_rows': 0})

    return JsonResponse({'columns': [], 'max_rows': 0})

# Function to reformat dd/mm/yyyy into yyyy-mm-dd datetime objects
def format_dates(df):
    date_cols = df.select_dtypes(include=['object']).columns
    for col in date_cols:
        # Check if the first entry matches the dd/mm/yyyy format
        if re.match(r'^\d{2}/\d{2}/\d{4}$', df[col].iloc[0]):
            # Convert entire column to datetime and reformat to yyyy-mm-dd
            df[col] = pd.to_datetime(df[col], format='%d/%m/%Y', errors='coerce').dt.strftime('%Y-%m-%d')
    
    return df

# Function to clean data by replacing erroneous values with zero
def clean_data(df):
    # Replace -9999 with zero
    df.replace(-9999, 0, inplace=True)
    # Handle invalid date formats
    print(f'{df.dtypes}')
    df = format_dates(df)
    # Replace missing values with zero
    df.fillna(0, inplace=True)
    return df

# Function that gets the columns from the db of type float
def get_float_columns(db):
    float_columns = []
    for field in db._meta.get_fields():
        if isinstance(field, models.FloatField):
            float_columns.append(field.name)
    return float_columns

# Function that creates a custom dataset as a dataframe
def create_custom_dataset(files, columns, start_row, end_row, feature_eng_choices):
    # Initialise an empty list to hold dataframes
    dataframes = []

    # Loop through each file and read it as DataFrame
    for file_path in files:
        full_path = os.path.join(settings.MEDIA_ROOT, file_path)
        chunk_iter = pd.read_csv(full_path, usecols=columns, chunksize=100000)
        
        for chunk in chunk_iter:
            dataframes.append(chunk)

    # Concatenate all the DataFrames
    concat_df = pd.concat(dataframes, ignore_index=True)

    # Trim the DataFrame to the specified rows
    trimmed_df = concat_df.iloc[start_row:end_row]

    # Clean the DataFrame by handling erroneous values
    if 'handle_missing' in feature_eng_choices:
        trimmed_df = clean_data(trimmed_df)
        print('DataFrame cleaned successfully.')

    # Fetches the columns which contain type float
    float_columns = [col for col in trimmed_df.columns if pd.api.types.is_float_dtype(trimmed_df[col]) or pd.api.types.is_integer_dtype(trimmed_df[col])]

    # Normalize the DataFrame
    if 'normalize' in feature_eng_choices:
        float_cols = trimmed_df.columns.intersection(float_columns)
        scaler = MinMaxScaler()
        trimmed_df[float_cols] = scaler.fit_transform(trimmed_df[float_cols])
        print('DataFrame normalized successfully.')

    # Standardize the DataFrame
    if 'standardize' in feature_eng_choices:
        float_cols = trimmed_df.columns.intersection(float_columns)
        scaler = StandardScaler()
        trimmed_df[float_cols] = scaler.fit_transform(trimmed_df[float_cols])
        print('DataFrame standardized successfully.')

    return trimmed_df

# Function that saves a dataset to a chosen db
def save_dataset_to_db(dataset, db):
    # Initializes batch size for entries saved to db
    batch_size = 500
    # Convert DataFrame rows to a list of model instances with progress printing
    entries = []
    total_rows = len(dataset)
    for index, row in dataset.iterrows():
        entries.append(db(**row.to_dict()))

        # Progress printing
        if (index + 1) % batch_size == 0 or (index + 1) == total_rows:
            print(f'Entries converted: {index + 1}/{total_rows}')

    # Checks if all the columns of the dataset match the db
    db_columns = [field.name for field in db._meta.get_fields()]
    dataset_columns = dataset.columns
    if all(column in db_columns for column in dataset_columns):
        try:
            # Save entries in batches
            with transaction.atomic():
                for i in range(0, len(entries), batch_size):
                    db.objects.bulk_create(entries[i:i+batch_size])
                    print(f'Entries committed: {i}/{len(entries)}.')
                print('Dataset saved successfully')
                return db
        except DatabaseError as e:
            print(f'An error occured while creating the table: {e}')
            return False
        except Exception as e:
            print(f'An unexpected error occured: {e}')
            return False

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

def get_db_file_path():
    db_path = settings.DATABASES['default']['NAME']
    return db_path

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

# Function that fetches the first 50 rows of a chosen database table
def fetch_sample_dataset(db, sample_size):
    db_data = db.objects.all().values()[:sample_size]  # Get the first 50 rows
    columns = db_data[0].keys() if db_data else []  # Get column names

    return db_data, columns

def populate_process_data_form(form):
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

# Function that generates process_data form and handles its logic
def process_data_form(request):
    # Populates the form
    if request.method == 'POST':
        form = ProcessDataForm(request.POST)
        form = populate_process_data_form(form)

        # Get the selected files
        selected_files = request.POST.getlist('files')
        if selected_files:
            common_columns = get_common_columns(selected_files)
            form.fields['columns'].choices = [(col, col) for col in common_columns]

        if form.is_valid():
            # Fetches form variables
            files, columns, dataset_type, start_row, end_row, feature_eng, title, comment = fetch_process_data_form_choices(form)

            # Attempts to create the dataset, create a db, and save dataset to db
            dataset = create_custom_dataset(files, columns, start_row, end_row, feature_eng)
            db = create_custom_db(title, dataset)
            dataset_saved = save_dataset_to_db(dataset, db)

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
        form = populate_process_data_form(form)

    return render(request, 'process_data_form.html', {'form': form})

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

# Function that handles the dynamic update for the build model form
def update_build_model_form(request):
    hidden_layers = int(request.POST.get('hidden_layers', 0))
    form = BuildModelForm()

    layer_html = ""
    for i in range(hidden_layers):
        if i < hidden_layers - 1:
            # Render Hidden layers
            layer_html += f'<div><label for="layer_type_{i}">Hidden Layer Type {i+1}</label>{form["layer_type"]}</div>'
            layer_html += f'<div><label for="nodes_{i}">Nodes {i+1}</label><input type="number" name="nodes_{i}" id="nodes_{i}" required></div>'
            layer_html += f'<div><label for="activation_{i}">Activation {i+1}</label>{form["activation"]}</div>'
        else:
            # Render Output layer
            layer_html += f'<div><label for="layer_type_{i}">Output Layer Type {i+1}</label>{form["layer_type"]}</div>'
            layer_html += f'<div><label for="nodes_{i}">Outputs {i+1}</label><input type="number" name="nodes_{i}" id="nodes_{i}" required></div>'
            layer_html += f'<div><label for="activation_{i}">Activation {i+1}</label>{form["activation"]}</div>'

    return JsonResponse({'layer_html': layer_html})

# Function that handles the build_model_form logic
def build_model_form(request):
    if request.method == 'POST':
        form = BuildModelForm(request.POST)
        
        if form.is_valid():
            print('form verified')
            title = form.cleaned_data['model_title']
            comment = form.cleaned_data['comment']
            features = form.cleaned_data['features']
            hidden_layers = form.cleaned_data['hidden_layers']
            optimizer = form.cleaned_data['optimizer']
            loss = form.cleaned_data['loss']
            metrics = form.cleaned_data['metrics']
            
            # Extract hidden layers data
            layer_types = []
            nodes = [features]
            activations = []
            for i in range(hidden_layers):
                layer_types.append(request.POST.get(f'layer_type_{i}'))
                nodes.append(int(request.POST.get(f'nodes_{i}')))
                activations.append(request.POST.get(f'activation_{i}'))
            
            user = request.user
            model = build_model(title, user, comment, layer_types, activations, nodes, optimizer, loss, metrics)
            print(f'model built successfully: {model.summary}')
            #save_model(title, comment, model)
            return redirect('home')
        else:
            print(f'form errors: {form.errors}')
            
    else:
        form = BuildModelForm()

    return render(request, 'build_model_form.html', {'form': form})

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

# Function that trains the selected model
def train_model(features, output, model, batch_size, epochs, verbose, validation_split):
        history = model.fit(features, output,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=verbose,
                        validation_split=validation_split)
        return history, model

def fetch_train_model_form_choices(form):
    features = form.cleaned_data['feature_dataset']
    output = form.cleaned_data['training_dataset']
    batch_size = form.cleaned_data['batch_size']
    epochs = form.cleaned_data['epochs']
    verbose = form.cleaned_data['verbose']
    validation_split = form.cleaned_data['validation_split']

    return features, output, batch_size, epochs, verbose, validation_split

# Function that populates the train model form choices
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

# Function that renders the train model form
def train_model_form(request):
    if request.method == 'POST':
        form = TrainModelForm(request.POST)
        if form.is_valid():
            features, output, model, batch_size, epochs, verbose, validation_split = fetch_train_model_form_choices(form)
            train_model(features, output, model, batch_size, epochs, verbose, validation_split)

            title = form.cleaned_data['title']
            comment = form.cleaned_data['comment']
            user = request.user
            save_model(title, model, 'trained', comment, user)
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