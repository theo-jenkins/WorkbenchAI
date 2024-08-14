from django.http import JsonResponse
from django.template.loader import render_to_string
from tensorflow.keras.models import load_model
from .forms import ProcessTimeSeriesForm, ProcessTabularForm, BuildSequentialForm
from .models import Metadata
from .site_functions import get_max_rows, get_common_columns
from .db_functions import get_input_shape


# Function that updates the base process data form depending if timeseries or tabular is selected
def update_process_data_form(request):
    dataset_form = request.POST.get('dataset_form')
    if dataset_form == 'ts':
        ts_form = ProcessTimeSeriesForm(feature_count=1, common_columns=[])
        form_html = render_to_string('datasets/process_ts_form.html', {'ts_form': ts_form}, request=request)
    elif dataset_form == 'tabular':
        tabular_form = ProcessTabularForm(feature_count=1, common_columns=[])
        form_html = render_to_string('datasets/process_tabular_form.html', {'tabular_form': tabular_form}, request=request)
    else:
        form_html = ''
    return JsonResponse({'process_dataset_form_html': form_html})

# Function to update the timeseries dataset form
def update_tabular_form(request):
    if request.method == 'POST':
        selected_files = request.POST.getlist('files[]')
        features = int(request.POST.get('features', 1))
        common_columns = get_common_columns(selected_files)
        max_rows = get_max_rows(selected_files)
        ts_form = ProcessTabularForm(feature_count=features, common_columns=common_columns)
        context = {
            'form': ts_form,
            'range': range(features)
        }
        feature_fields_html = render_to_string('partials/feature_fields.html', context)

        return JsonResponse({
                'columns': common_columns,
                'max_rows': max_rows,
                'feature_fields_html': feature_fields_html
        })

    return JsonResponse({'error': 'This endpoint only supports POST requests'}, status=400)

# Function to update the timeseries dataset form
def update_ts_form(request):
    if request.method == 'POST':
        selected_files = request.POST.getlist('files[]')
        features = int(request.POST.get('features', 1))
        common_columns = get_common_columns(selected_files)
        max_rows = get_max_rows(selected_files)
        tabular_form = ProcessTimeSeriesForm(feature_count=features, common_columns=common_columns)
        context = {
            'form': tabular_form,
            'range': range(features)
        }
        feature_fields_html = render_to_string('partials/feature_fields.html', context)

        return JsonResponse({
                'columns': common_columns,
                'max_rows': max_rows,
                'feature_fields_html': feature_fields_html
        })

    return JsonResponse({'error': 'This endpoint only supports POST requests'}, status=400)

# Fetches and retuns the process data form user choices
def fetch_process_data_form_choices(form):
    title = form.cleaned_data['db_title']
    comment = form.cleaned_data['comment']
    dataset_form = form.cleaned_data['dataset_form']
    dataset_type = form.cleaned_data['dataset_type']

    return title, comment, dataset_form, dataset_type

# Fetches and retunrs the tabular dataset form user choices
def fetch_tabular_form_choices(form):
    file_paths = form.cleaned_data['files']
    start_row = form.cleaned_data['start_row']
    end_row = form.cleaned_data['end_row']
    num_features = form.cleaned_data['features']
    
    features = []
    feature_eng_choices = []
    for i in range(num_features):
        features.append(form.cleaned_data[f'column_{i}'])
        feature_eng_choices.append(form.cleaned_data[f'feature_eng_{i}'])

    return file_paths, start_row, end_row, features, feature_eng_choices

# Fetches and returns the timeseries dataset form user choices
def fetch_ts_form_choices(form):
    aggregation_method = form.cleaned_data['aggregation_method']
    file_paths = form.cleaned_data['files']
    start_row = form.cleaned_data['start_row']
    end_row = form.cleaned_data['end_row']
    num_features = form.cleaned_data['features']
    
    features = []
    feature_eng_choices = []
    for i in range(num_features):
        features.append(form.cleaned_data[f'column_{i}'])
        feature_eng_choices.append(form.cleaned_data[f'feature_eng_{i}'])

    return aggregation_method, file_paths, start_row, end_row, features, feature_eng_choices



####################################################################

# Function that handles the select model type form update
def update_build_model_form(request):
    model_form = request.POST.get('model_form')

    if model_form == 'sequential':
        seq_form = BuildSequentialForm(hidden_layer_count=1)
        form_html = render_to_string('models/build_sequential_model_form.html', {'seq_form': seq_form}, request=request)
    elif model_form == 'xgboost':
        form_html = render_to_string('models/build_xgboost_model_form.html', {}, request=request)
    else:
        form_html = ''

    return JsonResponse({'model_form_html': form_html})

def update_sequential_model_form(request):
    hidden_layer_count = int(request.POST.get('hidden_layers', 1))
    seq_form = BuildSequentialForm(hidden_layer_count=hidden_layer_count)
    context = {
        'seq_form': seq_form,
        'range': range(hidden_layer_count)
    }
    hidden_layer_html = render_to_string('partials/hidden_layer_form.html', context)
    return JsonResponse({'hidden_layer_html': hidden_layer_html})

# Function that returns the user choices for the BuildModelForm
def fetch_build_model_form_choices(form):
    model_title = form.cleaned_data['model_title']
    model_comment = form.cleaned_data['comment']
    model_form = form.cleaned_data['model_form']
    dataset_id = form.cleaned_data['feature_dataset']

    return model_title, model_comment, model_form, dataset_id

# Function that handles the user choices for the sequential model form
def fetch_sequential_model_form_choices(form, feature_dataset):
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

    # Calculates the input shape for tabular or timeseries data
    input_shape = get_input_shape(feature_dataset)

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
    timesteps = form.cleaned_data['timesteps']

    return title, comment, features_id, outputs_id, model_id, batch_size, epochs, verbose, validation_split, timesteps

