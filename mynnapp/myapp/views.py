from django.core.exceptions import ValidationError
from django.conf import settings
from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate
from django.contrib.auth.decorators import login_required
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
def handle_uploaded_file(file):
    valid_extensions = ['.zip','.csv']
    if not any(file.name.endswith(ext) for ext in valid_extensions):
        raise ValidationError('Invalid file extension')
    
    if file.name.endswith('.csv'):
        try:
            # Ensure the uploaded_files directory exists
            if not os.path.exists(settings.MEDIA_ROOT):
                os.makedirs(settings.MEDIA_ROOT)
            #Saves the uploaded file to the directory
            file_path = os.path.join(settings.MEDIA_ROOT, file.name)
            with open(file_path, 'wb+') as destination:
                for chunk in file.chunks():
                    destination.write(chunk)
            return file, file_path
        except Exception as e:
            print(f"An error occurred while handling the file: {e}")
            raise ValidationError("There was an error uploading the file. Please try again.")

    
def upload_data(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES) #Creates the upload file form instance

        if form.is_valid():
            try:
                file, file_path = handle_uploaded_file(request.FILES['file'])
                request.session['last_file_path'] = file_path
                return render(request, 'home.html', {'file': file})
            except ValidationError as e:
                print(f"Error {e}")
                form.add_error('file', e)
    else:
        form = UploadFileForm()

    return render(request, 'upload_data.html', {'form': form})

def process_data(request):
    file_path = request.session.get('last_file_path')
    if not file_path:
        error_message = "file_path has not been found"
        return render(request, 'home.html', {'error_message': error_message})
    
    # Construct the full file path
    file_path = os.path.join(settings.MEDIA_ROOT, file_path)

    # Check if the file actually exists on the filesystem
    if not os.path.exists(file_path):
        error_message = "The uploaded file does not exist."
        return render(request, 'home.html', {'error_message': error_message})
    
    df = pd.read_csv(file_path)
    column_choices = [(col, col) for col in df.columns]

    if request.method == 'POST':
        form = ProcessDataForm(request.Post)
        form.fields['columns'].choices = column_choices

        if form.is_valid():
            selected_columns = form.cleaned_data['columns']
            start_row = form.cleaned_data['star_row']
            end_row = form.cleaned_data['end_row']
            # Process the data based on user input
            processed_df = df.loc[start_row:end_row, selected_columns]
            # You can now handle the processed_df as needed
            return render(request, 'process_success.html', {'data': processed_df.to_html()})
    else:
        form = ProcessDataForm()
        form.fields['columns'].choices = column_choices

    return render(request, 'process_data.html', {'form': form, 'file': os.path.basename(file_path)})