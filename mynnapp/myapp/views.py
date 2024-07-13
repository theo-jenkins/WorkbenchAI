import pandas as pd
import tensorflow as tf
import os
import zipfile
import re
from django.core.exceptions import ValidationError
from django.conf import settings
from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.db import DatabaseError, transaction, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from .forms import UploadFileForm, CustomUserCreationForm, ProcessDataForm, BuildModelForm
from .models import create_custom_db

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
                zip_ref.extractall(upload_dir)
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

# Function that creates a custom dataset and commits to the db
def create_custom_dataset(form, db):
    selected_files = form.cleaned_data['files']
    selected_columns = form.cleaned_data['columns']
    start_row = form.cleaned_data['start_row']
    end_row = form.cleaned_data['end_row']
    feature_eng_options = form.cleaned_data['feature_eng']

    # Initialise an empty list to hold dataframes
    dataframes = []
    batch_size = 500

    # Loop through each file and read it as DataFrame
    for file_path in selected_files:
        full_path = os.path.join(settings.MEDIA_ROOT, file_path)
        df = pd.read_csv(full_path, usecols=selected_columns)
        dataframes.append(df)
    
    # Concatenate all the DataFrames
    concat_df = pd.concat(dataframes, ignore_index=True)

    # Trim the DataFrame to the specified rows
    trimmed_df = concat_df.loc[start_row:end_row]

    # Clean the DataFrame by handling erroneous values
    if 'handle_missing' in feature_eng_options:
        trimmed_df = clean_data(trimmed_df)
        print('DataFrame cleaned successfully.')

    # Fetches the float columns from the db
    float_columns = get_float_columns(db)

    # Normalize the DataFrame
    if 'normalize' in feature_eng_options:
        float_cols = trimmed_df.columns.intersection(float_columns)
        scaler = MinMaxScaler()
        trimmed_df[float_cols] = scaler.fit_transform(trimmed_df[float_cols])
        print('DataFrame normalized successfully.')

    # Standardize the DataFrame
    if 'standardize' in feature_eng_options:
        float_cols = trimmed_df.columns.intersection(float_columns)
        scaler = StandardScaler()
        trimmed_df[float_cols] = scaler.fit_transform(trimmed_df[float_cols])
        print('DataFrame standardized successfully.')

    # Convert DataFrame rows to a list of model instances
    entries = [
        db(**row.to_dict())
        for index, row in trimmed_df.iterrows()
    ]

    try:
        # Save entries in batches
        with transaction.atomic():
            for i in range(0, len(entries), batch_size):
                db.objects.bulk_create(entries[i:i+batch_size])
                print(f'Entries saved: {i}/{len(entries)}.')
            print('Dataset saved successfully')
            return db
    except DatabaseError as e:
        print(f'An error occured while creating the table: {e}')
        return False
    except Exception as e:
        print(f'An unexpected error occured: {e}')
        return False

# Function that fetches a sample row from the .csv file
def get_sample_row(selected_files):
    for file in selected_files:
        full_path = os.path.join(settings.MEDIA_ROOT, file)
        df = pd.read_csv(full_path)
        sample_row = df.dropna().iloc[0] # Selects the top row
        if not sample_row.isnull().values.any():
            return sample_row
        else:
            print('Could not fetch sample row.')
    return None

def fetch_process_data_form_choices():
    file_list = get_uploaded_files()
    feature_eng_choices = [
        ('handle_missing', 'Handle missing values'), 
        ('normalize', 'Normalization'), 
        ('standardize', 'Standardization')
    ]
    return file_list, feature_eng_choices   

# Function that generates process_data form and handles its logic
def process_data_form(request):
    file_list, feature_eng_choices = fetch_process_data_form_choices()
    
    # Populates the form
    if request.method == 'POST':
        form = ProcessDataForm(request.POST)
        form.fields['feature_eng'].choices = feature_eng_choices
        form.fields['files'].choices = file_list

        # Get the selected files
        selected_files = request.POST.getlist('files')
        if selected_files:
            common_columns = get_common_columns(selected_files)
            form.fields['columns'].choices = [(col, col) for col in common_columns]

        if form.is_valid():
            # Attempts to create the custom db
            title = form.cleaned_data['db_title']
            columns = form.cleaned_data['columns']
            sample_row = get_sample_row(selected_files)
            db = create_custom_db(title, columns, sample_row)

            if db:
                db_columns = [field.name for field in db._meta.get_fields()]
                if all(column in db_columns for column in columns):
                    dataset_created = create_custom_dataset(form, db)
                    if dataset_created:
                        db_data = db.objects.all().values()[:50]  # Get the first 50 rows
                        columns = db_data[0].keys() if db_data else []  # Get column names
                        return render(request, 'sample_dataset.html', {'title': title, 'db_data': db_data, 'columns': columns})
    else:
        form = ProcessDataForm()
        form.fields['feature_eng'].choices = feature_eng_choices
        form.fields['files'].choices = file_list

    return render(request, 'process_data_form.html', {'form': form})

def process_success(request):
    return render(request, 'process_success.html')

# Function that builds a fully connected neural network
def build_model(title, model_type, features, hidden_layers, outputs, optimizer, loss, metrics):
    model = Sequential()

    if model_type == 'fully_connected':
        model.add(Dense(64, input_dim=features, activation='relu'))  # Input layer

        # Add hidden layers
        for _ in range(hidden_layers):
            model.add(Dense(64, activation='relu'))

        # Output layer
        model.add(Dense(outputs))

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Saves the model as a file
    upload_dir = os.path.join(settings.BASE_DIR, 'nn_models')
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    model_path = os.path.join(upload_dir, f'{title}.keras')
    model.save(model_path)

    return model

# Function that fetches all the neural network options for the keras library
def fetch_keras_choices():
    model_type_choices = [
        ('fully_connected', 'Fully connected'),
        ('convolutional', 'Convolutional'),
        ('recurrent', 'Recurrent')
    ]
    optimizer_choices = [
        ('adam', 'Adam'),
        ('sgd', 'SGD'),
        ('rmsprop', 'RMSprop'),
        ('adagrad', 'Adagrad'),
        ('adadelta', 'Adadelta'),
        ('adamax', 'Adamax'),
        ('nadam', 'Nadam')
    ]
    loss_choices = [
        ('categorical_crossentropy', 'Categorical Cross-Entropy'),
        ('binary_crossentropy', 'Binary Cross-Entropy'),
        ('mean_squared_error', 'Mean Squared Error'),
        ('mean_absolute_error', 'Mean Absolute Error'),
        ('hinge', 'Hinge Loss'),
        ('sparse_categorical_crossentropy', 'Sparse Categorical Cross-Entropy'),
    ]
    metric_choices = [
        ('accuracy', 'Accuracy'),
        ('precision', 'Precision'),
        ('recall', 'Recall'),
        ('f1_score', 'F1 Score'),
        ('mean_squared_error', 'Mean Squared Error'),
        ('mean_absolute_error', 'Mean Absolute Error')
    ]
    return model_type_choices, optimizer_choices, loss_choices, metric_choices

# Function that handles the build_model_form logic
def build_model_form(request):
    model_type_choices, optimizer_choices, loss_choices, metric_choices = fetch_keras_choices()

    if request.method == 'POST':
        form = BuildModelForm(request.POST)
        form.fields['model_type'].choices = model_type_choices
        form.fields['optimizer'].choices = optimizer_choices
        form.fields['loss'].choices = loss_choices
        form.fields['metrics'].choices = metric_choices
        
        if form.is_valid():
            title = form.cleaned_data['model_title']
            model_type = form.cleaned_data['model_type']
            features = form.cleaned_data['features']
            hidden_layers = form.cleaned_data['hidden_layers']
            outputs = form.cleaned_data['outputs']
            optimizer = form.cleaned_data['optimizer']
            loss = form.cleaned_data['loss']
            metrics = form.cleaned_data['metrics']
            
            model = build_model(title, model_type, features, hidden_layers, outputs, optimizer, loss, metrics)

            return redirect('home')
    else:
        form = BuildModelForm()
        form.fields['model_type'].choices = model_type_choices
        form.fields['optimizer'].choices = optimizer_choices
        form.fields['loss'].choices = loss_choices
        form.fields['metrics'].choices = metric_choices

    return render(request, 'build_model_form.html', {'form': form})

def train_model_form(request):
    return render(request, 'train_model_form.html')