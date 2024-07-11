import pandas as pd
import os
import zipfile
from django.core.exceptions import ValidationError
from django.conf import settings
from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.db import DatabaseError, transaction, models
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from .forms import UploadFileForm, CustomUserCreationForm, ProcessDataForm
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
        try:
            df[col] = pd.to_datetime(df[col], format='%d/%m%Y', errors='coerce')
        except ValueError:
            pass

# Function to clean data by replacing erroneous values with zero
def clean_data(df):
    # Replace -9999 with zero
    df.replace(-9999, 0, inplace=True)
    # Handle invalid date formats
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
    trimmed_df = concat_df.iloc[start_row:end_row]

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
        

# Function that generates process_data form and handles its logic
def process_data_form(request):
    file_list = get_uploaded_files()
    feature_eng_choices = [
        ('handle_missing', 'Handle missing values'), 
        ('normalize', 'Normalization'), 
        ('standardize', 'Standardization')
    ]
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

            # Attempts the save the custom dataset
            if db:
                db_columns = [field.name for field in db._meta.get_fields()]
                if all(column in db_columns for column in columns):
                    if create_custom_dataset(form, db):
                        return redirect('process_success')
    else:
        form = ProcessDataForm()
        form.fields['feature_eng'].choices = feature_eng_choices
        form.fields['files'].choices = file_list

    return render(request, 'process_data_form.html', {'form': form})

def process_success(request):
    return render(request, 'process_success.html')