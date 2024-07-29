import requests
from django.core.exceptions import ValidationError
from django.conf import settings
from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from .forms import UploadFileForm, CustomUserCreationForm, ProcessDataForm, TrainModelForm, ProcessTickerForm, BuildModelForm
from .models import create_custom_db
from .tasks import create_custom_dataset, create_model_instances, train_model
from .site_functions import upload_file, get_uploaded_files, get_common_columns
from .db_functions import get_db_file_path, fetch_sample_dataset, save_metadata
from .model_functions import build_model, save_model
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
            file_paths, dataset_type, start_row, end_row, columns, feature_eng, title, comment = fetch_process_data_form_choices(form)

            # Attempts to create the dataset, create a db, and save dataset to db
            dataset = create_custom_dataset(file_paths, columns, start_row, end_row, feature_eng)
            db = create_custom_db(title, dataset)
            dataset_saved = create_model_instances(dataset, db)

            # Handles dataset saved successfully
            if dataset_saved:
                user = request.user
                file_path = get_db_file_path()
                metadata = save_metadata(title, comment, user, file_path, dataset_type)
                db_data, columns = fetch_sample_dataset(db, 50)
                return render(request, 'sample_dataset.html', {'title': title, 'db_data': db_data, 'columns': columns})
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
        form=BuildModelForm(hidden_layer_count=hidden_layer_count)

    layer_count_range = range(hidden_layer_count)
    return render(request, 'build_model_form.html', {'form': form, 'layer_count_range': layer_count_range})


# View that handles the train_model form
# Functions: fetch_train_model_form_choices(), populate_train_model_form()
def train_model_form(request):
    if request.method == 'POST':
        form = TrainModelForm(request.POST)
        if form.is_valid():
            features, output, model, batch_size, epochs, verbose, validation_split = fetch_train_model_form_choices(form)
            result = train_model(features, output, model, batch_size, epochs, verbose, validation_split)
            if result.successful():
                print(result.result)
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
