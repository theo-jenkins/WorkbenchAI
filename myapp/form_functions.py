from django.http import JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse
from django.template.loader import render_to_string
from .forms import ProcessDataForm, ProcessTimeSeriesForm, ProcessTabularForm, BuildModelForm, BuildSequentialForm, TrainModelForm, TrainTSModelForm, TrainTabularModelForm
from .models import Metadata, create_custom_db
from .site_functions import get_max_rows, get_common_columns
from .db_functions import get_db_file_path, get_input_shape, fetch_sample_dataset, save_metadata, prepare_datasets
from .model_functions import build_sequential_model, prepare_model, save_sequential_model, plot_metrics
from .tasks import create_custom_dataset, create_model_instances, train_model


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

# Function to handle dataset processing and saving for ProcessDataForm
def process_and_save_dataset(form, dataset_title, dataset_comment, dataset_form, dataset_type, request):
    feature_count = int(request.POST.get('features', 1))
    selected_files = request.POST.getlist('files')
    common_columns = get_common_columns(selected_files)

    # Check if the dataset form is for tabular data
    if dataset_form == 'tabular':
        form = ProcessTabularForm(request.POST, feature_count=feature_count, common_columns=common_columns)
        aggregation_method = None # No aggreagation needed for tabular data
    
    # Check if the dataset form is for time series data
    elif dataset_form == 'ts':
        form = ProcessTimeSeriesForm(request.POST, feature_count=feature_count, common_columns=common_columns)
    else:
        return None
    
    if form.is_valid():
        if dataset_form == 'ts': # Handle timeseries form choices
            aggregation_method, file_paths, start_row, end_row, features, feature_eng_choices = fetch_ts_form_choices(form)
        else: # Handle tabular form choices
            file_paths, start_row, end_row, features, feature_eng_choices = fetch_tabular_form_choices(form)

        # Create a DataFrame from the provided files and parameters
        dataset = create_custom_dataset(file_paths, features, start_row, end_row, feature_eng_choices, aggregation_method)

        # Create a database table from the DataFrame
        db = create_custom_db(dataset_title, dataset)

        # Populate the table with the table
        dataset_saved = create_model_instances(dataset, db)

        # Saves metadata entry if db table is created
        if db:
            user = request.user
            file_path = get_db_file_path()
            save_metadata(dataset_title, dataset_comment, user, file_path, dataset_form, dataset_type)
        
        # Renders dataset sample if data is saved
        if dataset_saved:
            data, features = fetch_sample_dataset(dataset_title, 50)
            return render(request, 'datasets/sample_dataset.html', {'title': dataset_title, 'data': data, 'features': features})
        else:
            print(f'Dataset not saved: {dataset_title}')
            return None


# Function that handles the logic for the process data form
def handle_process_data_form(request):
    if request.method == 'POST':
        form = ProcessDataForm(request.POST)
        if form.is_valid():
            # Extract choices made by the user in the form
            dataset_title, dataset_comment, dataset_form, dataset_type = fetch_process_data_form_choices(form)

            # Call the helper function to process and save the dataset
            response = process_and_save_dataset(form, dataset_title, dataset_comment, dataset_form, dataset_type, request)
            if response:
                return response
        else:
            form = ProcessDataForm()

    return render(request, 'datasets/process_data_form.html', {'form': form})


####################################################################

# Function that handles the select model type form update
def update_build_model_form(request):
    build_model_form = request.POST.get('build_model_form')

    if build_model_form == 'sequential':
        seq_form = BuildSequentialForm(hidden_layer_count=1)
        form_html = render_to_string('models/build_models/build_sequential_model_form.html', {'seq_form': seq_form}, request=request)
    elif build_model_form == 'xgboost':
        form_html = render_to_string('models/build_models/build_xgboost_model_form.html', {}, request=request)
    else:
        form_html = ''

    return JsonResponse({'build_model_form_html': form_html})

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

# Function that handles the logic for the build model form
def handle_build_model_form(request):
    if request.method == 'POST':
        form = BuildModelForm(request.POST)
        if form.is_valid():
            model_title, model_comment, model_form, dataset_id = fetch_build_model_form_choices(form)
            
            if model_form == 'sequential':
                hidden_layer_count = int(request.POST.get('hidden_layers', 1))
                seq_form = BuildSequentialForm(request.POST, hidden_layer_count=hidden_layer_count)
                
                if seq_form.is_valid():
                    input_shape, nodes, layer_types, activations, optimizer, loss, metrics = fetch_sequential_model_form_choices(seq_form, dataset_id)
                    user = request.user
                    
                    model = build_sequential_model(model_title, user, model_comment, layer_types, input_shape, nodes, activations, optimizer, loss, metrics)
                    if model:
                        model_metadata = Metadata.objects.get(title=model_title)
                        return redirect(reverse('view_model', kwargs={'model_id': model_metadata.id}))
                
                # If sequential form is invalid
                return render(request, 'models/build_models/build_model_form.html', {'form': form, 'form_errors': seq_form.errors})
        
        # If the build model form is invalid
        return render(request, 'models/build_models/build_model_form.html', {'form': form, 'form_errors': form.errors})

    # GET request: Display the empty build model form
    form = BuildModelForm()
    return render(request, 'models/build_models/build_model_form.html', {'form': form})

#########################################################################

# Function that returns all the user choices for the train model form
def fetch_train_model_form_choices(form):
    title = form.cleaned_data['model_title']
    comment = form.cleaned_data['comment']
    features_id = form.cleaned_data['feature_dataset']
    outputs_id = form.cleaned_data['training_dataset']
    model_id = form.cleaned_data['model']

    return title, comment, features_id, outputs_id, model_id

# Function that returns user choices for training model on tabular datasets
def fetch_tabular_train_model_form_choices(form):
    batch_size = form.cleaned_data['batch_size']
    epochs = form.cleaned_data['epochs']
    verbose = form.cleaned_data['verbose']
    validation_split = float(form.cleaned_data['validation_split'])

    return batch_size, epochs, verbose, validation_split

# Function that returns user choices for training model on timeseries datasets
def fetch_ts_train_model_form_choices(form):
    print(f'{form.cleaned_data}')
    batch_size = form.cleaned_data['batch_size']
    epochs = form.cleaned_data['epochs']
    verbose = form.cleaned_data['verbose']
    validation_split = float(form.cleaned_data['validation_split'])
    timesteps = form.cleaned_data['timesteps']

    return batch_size, epochs, verbose, validation_split, timesteps

# Function that updates the train model form depending on which datasets are selected
def update_train_model_form(request):
    feature_dataset_id = request.POST.get('feature_dataset')
    training_dataset_id = request.POST.get('training_dataset')
    model_id = request.POST.get('model')

    # Ensure all required fields are present
    if not (feature_dataset_id and training_dataset_id and model_id):
        return JsonResponse({'error': 'All datasets and model must be selected.'})

    feature_metadata = get_object_or_404(Metadata, id=feature_dataset_id)
    training_metadata = get_object_or_404(Metadata, id=training_dataset_id)
    model_metadata = get_object_or_404(Metadata, id=model_id)

    # Determine which form to load based on dataset types
    if feature_metadata.form == training_metadata.form:
        if feature_metadata.form == 'ts':
            train_model_form = TrainTSModelForm()
            template_name = 'models/train_models/train_ts_model_form.html'
        elif feature_metadata.form == 'tabular':
            train_model_form = TrainTabularModelForm()
            template_name = 'models/train_models/train_tabular_model_form.html'
        else:
            return JsonResponse({'error': 'Unsupported dataset form type.'})

        # Render the form to HTML
        form_html = render_to_string(template_name, {'form': train_model_form}, request=request)
        return JsonResponse({'train_model_form_html': form_html})

    else:
        # If dataset forms do not match, return an informative error message
        error_message = (
            'Please select datasets with matching forms. '
            f'Feature dataset form: {feature_metadata.form}, '
            f'Training dataset form: {training_metadata.form}. '            
        )
        return JsonResponse({'error': error_message})

# Function that handles the logic for the train model form
def handle_train_model_form(request):
    if request.method == 'POST':
        form = TrainModelForm(request.POST)
        if form.is_valid():
            # Fetches all user choices
            title, comment, feature_dataset_id, training_dataset_id, model_id = fetch_train_model_form_choices(form)

            feature_metadata = get_object_or_404(Metadata, id=feature_dataset_id)
            training_metadata = get_object_or_404(Metadata, id=training_dataset_id)

            if feature_metadata.form == training_metadata.form:
                if feature_metadata.form == 'tabular':
                    train_tabular_model_form = TrainTabularModelForm(request.POST)
                    if train_tabular_model_form.is_valid():
                        batch_size, epochs, verbose, validation_split = fetch_tabular_train_model_form_choices(train_tabular_model_form)
                        timesteps = None
                elif feature_metadata.form == 'ts':
                    train_ts_model_form = TrainTSModelForm(request.POST)
                    if train_ts_model_form.is_valid():
                        batch_size, epochs, verbose, validation_split, timesteps = fetch_ts_train_model_form_choices(train_ts_model_form)


            # Loads datasets and keras models
            features, outputs = prepare_datasets(feature_dataset_id, training_dataset_id, timesteps)
            model = prepare_model(model_id)

            # Trains model on loaded datasets
            history, model = train_model(features, outputs, model, batch_size, epochs, verbose, validation_split)

            # Saves model and history as a .png
            user = request.user
            save_sequential_model(title, model, history, 'sequential', 'trained', user, comment)
            fig_url = plot_metrics(title, history.history)

            # Render the evaluation page with the plot
            return render(request, 'models/evaluate_model.html', {'fig_url': fig_url})
    else:
        form = TrainModelForm()

    return render(request, 'models/train_models/train_model_form.html', {'form': form})

