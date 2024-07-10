from django.core.exceptions import ValidationError
from django.conf import settings
from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
import pandas as pd
import os
import zipfile
from .forms import UploadFileForm, CustomUserCreationForm, ProcessDataForm

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

# Function that finds the common columns between .csv files
def get_common_columns(file_paths):
    upload_dir = settings.MEDIA_ROOT
    full_paths = []

    # Traverse the directory to find all files
    for root, _, files in os.walk(upload_dir):
        for file in files:
            if file in file_paths:
                full_path = os.path.join(root, file)
                full_paths.append(full_path)

    # Reads the .csv and finds the common column headers
    dfs = [pd.read_csv(full_path) for full_path in full_paths]
    common_columns = set(dfs[0].columns)
    for df in dfs[1:]:
        common_columns.intersection_update(df.columns)
    return sorted(list(common_columns))

# Function that checks what files are selected and updates the form dynamically
def fetch_common_columns(request):
    if request.method == 'POST':
        selected_files = request.POST.getlist('files[]')
        common_columns = get_common_columns(selected_files)
        print(f'selected_files: {selected_files}')
        print(f'common_columns: {common_columns}')
        return JsonResponse({'columns': common_columns})
    return JsonResponse({'columns': []})

def process_data_form(request):
    form = ProcessDataForm()
    file_list = get_uploaded_files()
    form.fields['files'].choices = file_list

    return render(request, 'process_data_form.html', {'form': form})

def process_data(request):
    selected_files = request.session.get('selected_files', [])
    selected_columns = request.session.get('selected_columns', [])

    dfs = [pd.read_csv(os.path.join(settings.MEDIA_ROOT, file_path)) for file_path in selected_files]
    processed_dfs = [df[selected_columns] for df in dfs]
    combined_df = pd.concat(processed_dfs)

    return render(request, 'process_success.html', {'data': combined_df.to_html()})

def process_success(request):
    return render(request, 'process_success.html')