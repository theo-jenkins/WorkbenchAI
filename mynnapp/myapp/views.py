from django.core.exceptions import ValidationError
from django.conf import settings
from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
import pandas as pd
import os
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

#This functions saves the file and returns the file object
def upload_file(file):
    valid_extensions = ['.csv']
    
    # Check if the file has a valid extension
    if not any(file.name.endswith(ext) for ext in valid_extensions):
        raise ValidationError('Invalid file extension')
    
    try:
        # Ensure the uploaded_files directory exists
        if not os.path.exists(settings.MEDIA_ROOT):
            os.makedirs(settings.MEDIA_ROOT)
        
        # Save the uploaded file to the directory
        file_path = os.path.join(settings.MEDIA_ROOT, file.name)
        with open(file_path, 'wb+') as destination:
            for chunk in file.chunks():
                destination.write(chunk)
                
        return file  # Return the file object for further use if needed
    except Exception as e:
        print(f"An error occurred while handling the file: {e}")
        raise ValidationError("There was an error uploading the file. Please try again.")
    
def upload_data_form(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES) #Creates the upload file form instance

        if form.is_valid():
            try:
                file = request.FILES['file']
                upload_file(file)
                return render(request, 'home.html')
            except ValidationError as e:
                print(f"Error {e}")
                form.add_error('file', e)
    else:
        form = UploadFileForm()

    return render(request, 'upload_data_form.html', {'form': form})

def get_uploaded_files():
    folder_path = settings.MEDIA_ROOT
    if not os.path.exists(folder_path) or not os.listdir(folder_path):
        return []
    return [(file, file) for file in os.listdir(folder_path)]

def get_common_columns(file_paths):
    dfs = [pd.read_csv(os.path.join(settings.MEDIA_ROOT, file_path)) for file_path in file_paths]
    common_columns = set(dfs[0].columns)
    for df in dfs[1:]:
        common_columns.intersection_update(df.columns)
    print(f'columns:{common_columns}')
    return sorted(list(common_columns))

def fetch_common_columns(request):
    if request.method == 'POST':
        selected_files = request.POST.getlist('files[]')
        common_columns = get_common_columns(selected_files)
        return JsonResponse({'columns': common_columns})
    return JsonResponse({'columns': []})

def process_data_form(request):
    file_choices = get_uploaded_files()
    form = ProcessDataForm()
    form.fields['files'].choices = file_choices

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