import sqlite3
from django.http import JsonResponse
from django.template.loader import render_to_string
from keras.models import load_model
import pandas as pd
from .forms import ProcessDataForm, BuildModelForm
from .models import Metadata
from .site_functions import get_max_rows, get_common_columns, get_uploaded_files

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

# Function that handles the dynamic update for the build model form
def update_build_model_form(request):
    if request.method =='POST':
        hidden_layer_count = int(request.POST.get('hidden_layers', 1))
        form = BuildModelForm(hidden_layer_count=hidden_layer_count)
        context = {
            'form': form,
            'range': range(hidden_layer_count)
        }
        layer_html = render_to_string('layer_fields.html', context)
        return JsonResponse({'layer_html': layer_html})
    return JsonResponse({'error': 'Invalid request'}, status=400)

# Function that handles the user choices for the build model form
def process_build_model_form(form):
    title = form.cleaned_data['model_title']
    comment = form.cleaned_data['comment']
    features = form.cleaned_data['features']
    feature_layer_type = form.cleaned_data['feature_layer_type']
    feature_activation = form.cleaned_data['feature_activation']
    outputs = form.cleaned_data['outputs']
    output_layer_type = form.cleaned_data['output_layer_type']
    output_activation = form.cleaned_data['output_activation']
    optimizer = form.cleaned_data['optimizer']
    loss = form.cleaned_data['loss']
    metrics = form.cleaned_data['metrics']
    hidden_layer_count = form.cleaned_data['hidden_layers']

    # Processes the form data into lists for the build_model() function
    nodes = [features]
    layer_types = [feature_layer_type]
    activations = [feature_activation]
    for i in range(hidden_layer_count):
        nodes.append(form.cleaned_data[f'nodes_{i}'])
        layer_types.append(form.cleaned_data[f'layer_type_{i}'])
        activations.append(form.cleaned_data[f'activation_{i}'])
    nodes.append(outputs)
    layer_types.append(output_layer_type)
    activations.append(output_activation)

    return title, comment, nodes, layer_types, activations, optimizer, loss, metrics

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

# Function to retrieve a table from the db and return as df
def load_sqlite_table(db_path, table_name):
    conn = sqlite3.connect(db_path)
    query = f'SELECT * FROM "myapp_{table_name}"'
    df = pd.read_sql_query(query, conn)
    # Drop the first column (ID column)
    df = df.iloc[:, 1:]
    conn.close()
    return df