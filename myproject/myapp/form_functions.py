import sqlite3
from django.http import JsonResponse
from django.template.loader import render_to_string
from keras.models import load_model
import pandas as pd
from .forms import ProcessDataForm, BuildModelForm, BuildSequentialForm
from .models import Metadata
from .site_functions import get_max_rows, get_common_columns
from .db_functions import load_sqlite_table, calc_dataset_shape

# Function that checks what files are selected and updates the form dynamically
def update_process_data_form(request):
    if request.method == 'POST':
        selected_files = request.POST.getlist('files[]')
        features = int(request.POST.get('features', 1))
        common_columns = get_common_columns(selected_files)
        max_rows = get_max_rows(selected_files)
        form = ProcessDataForm(feature_count=features, common_columns=common_columns)       

        context = {
            'form': form,
            'range': range(features)
        }
        feature_fields_html = render_to_string('feature_fields.html', context)

        return JsonResponse({
            'columns': common_columns,
            'max_rows': max_rows,
            'feature_fields_html': feature_fields_html
        })

    return JsonResponse({'error': 'This endpoint only supports POST requests'}, status=400)

# Fetches and returns all the user responses for the process_data form
def fetch_process_data_form_choices(form):
    file_paths = form.cleaned_data['files']
    dataset_type = form.cleaned_data['dataset_type']
    start_row = form.cleaned_data['start_row']
    end_row = form.cleaned_data['end_row']
    title = form.cleaned_data['db_title']
    comment = form.cleaned_data['comment']

    features_num = form.cleaned_data['features']
    features = []
    feature_eng_choices = []
    for i in range(features_num):
        features.append(form.cleaned_data[f'column_{i}'])
        feature_eng_choices.append(form.cleaned_data[f'feature_eng_{i}'])

    return file_paths, dataset_type, start_row, end_row, features, feature_eng_choices, title, comment


####################################################################

# Function that handles the select model type form update
def update_build_model_form(request):
    model_type = request.POST.get('model_type')

    if model_type == 'sequential':
        seq_form = BuildSequentialForm(hidden_layer_count=1)
        form_html = render_to_string('models/build_sequential_model_form.html', {'seq_form': seq_form}, request=request)
    elif model_type == 'xgboost':
        form_html = render_to_string('models/build_xgboost_model_form.html', {}, request=request)
    else:
        form_html = ''

    return JsonResponse({'model_form_html': form_html})


def update_sequential_model_form(request):
    hidden_layer_count = int(request.POST.get('hidden_layers', 1))
    form = BuildSequentialForm(hidden_layer_count=hidden_layer_count)
    context = {
        'form': form,
        'range': range(hidden_layer_count)
    }
    hidden_layer_html = render_to_string('partials/hidden_layer_form.html', context)
    return JsonResponse({'hidden_layer_html': hidden_layer_html})

# Function that handles the user choices for the sequential model form
def process_sequential_model_form(form, feature_dataset):
    # Fetches the user choices from the form
    input_nodes = form.cleaned_data['input_nodes']
    input_layer_type = form.cleaned_data['input_layer_type']
    input_activation = form.cleaned_data['input_activation']
    outputs = form.cleaned_data['outputs']
    output_layer_type = form.cleaned_data['output_layer_type']
    output_activation = form.cleaned_data['output_activation']
    optimizer = form.cleaned_data['optimizer']
    loss = form.cleaned_data['loss']
    metrics = form.cleaned_data['metrics']
    hidden_layer_count = form.cleaned_data['hidden_layers']

    # Processes the form data into lists for the build_model() function
    print(f'Form cleaned_data: {form.cleaned_data}')
    nodes = [input_nodes]
    layer_types = [input_layer_type]
    activations = [input_activation]
    for i in range(hidden_layer_count):

        nodes.append(form.cleaned_data[f'nodes_{i}'])
        layer_types.append(form.cleaned_data[f'layer_type_{i}'])
        activations.append(form.cleaned_data[f'activation_{i}'])
    nodes.append(outputs)
    layer_types.append(output_layer_type)
    activations.append(output_activation)

    # Calculates the number of features from the selected dataset
    shape = calc_dataset_shape(feature_dataset)
    if not shape:
        raise ValueError(f'An error occured calculated the shape of the dataset: {feature_dataset}')
    input_shape = (shape[1],)

    return input_shape, nodes, layer_types, activations, optimizer, loss, metrics

#########################################################################

# Function that returns all the user choices for the train model form
def fetch_train_model_form_choices(form):
    title = form.cleaned_data['model_title']
    comment = form.cleaned_data['comment']
    features_id = form.cleaned_data['feature_dataset']
    outputs_id = form.cleaned_data['training_dataset']
    model_id = form.cleaned_data['model']
    batch_size = form.cleaned_data['batch_size']
    epochs = form.cleaned_data['epochs']
    verbose = form.cleaned_data['verbose']
    validation_split = float(form.cleaned_data['validation_split'])

    # Retrieves the Metadata objects
    try:
        features_metadata = Metadata.objects.get(id=features_id)
        outputs_metadata = Metadata.objects.get(id=outputs_id)
        model_metadata = Metadata.objects.get(id=model_id)
    except Metadata.DoesNotExist:
        return None, None, None, batch_size, epochs, verbose, validation_split
    
    # Load features and outputs from SQLite database
    features, outputs = None, None
    try:
        features = load_sqlite_table(features_metadata.file_path, features_metadata.title)
        outputs = load_sqlite_table(outputs_metadata.file_path, outputs_metadata.title)
    except Exception as e:
        print(f'Error loading SQLite tables: {e}')
    
    # Load keras model
    model = None
    try:
        model = load_model(model_metadata.file_path)
    except Exception as e:
        print(f'Error loading keras model: {e}')

    return title, comment, features, outputs, model, batch_size, epochs, verbose, validation_split

