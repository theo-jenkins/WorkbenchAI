import os
from django.http import JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse
from django.template.loader import render_to_string
from django.conf import settings
from .forms import ProcessDataForm, ProcessTimeSeriesForm, BuildModelForm, BuildSequentialForm, TrainModelForm, TrainTSModelForm, TrainTabularModelForm, MakePredictionForm
from .models import FileMetadata, DatasetMetadata, ModelMetadata, create_custom_db
from .site_functions import get_max_rows, get_common_columns
from .db_functions import get_db_file_path, fetch_sample_dataset, prepare_datasets, save_dataset_metadata
from .model_functions import build_sequential_model, prepare_model, save_sequential_model, plot_metrics, load_features, prepare_features, get_max_version
from .tasks import create_custom_dataset, create_model_instances, train_model

# Fetches and retuns the process data form user choices
def fetch_process_data_form_choices(form):
    title = form.cleaned_data['dataset_title']
    comment = form.cleaned_data['comment']
    dataset_form = form.cleaned_data['dataset_form']
    dataset_type = form.cleaned_data['dataset_type']
    file_ids = form.cleaned_data['files']
    start_row = form.cleaned_data['start_row']
    end_row = form.cleaned_data['end_row']
    num_features = form.cleaned_data['features']

    features = []
    feature_eng_choices = []
    for i in range(num_features):
        features.append(form.cleaned_data[f'column_{i}'])
        feature_eng_choices.append(form.cleaned_data[f'feature_eng_{i}'])

    return title, comment, dataset_form, dataset_type, file_ids, start_row, end_row, features, feature_eng_choices

# Fetches and returns the timeseries dataset form user choices
def fetch_ts_form_choices(form):
    aggregation_method = form.cleaned_data['aggregation_method']

    return aggregation_method

# Populates the process data form with relevant form choices
def update_process_data_form(request):
    if request.method == 'POST':
        # Extract the selected files and features count from the POST data
        selected_files = request.POST.getlist('files[]')
        features = int(request.POST.get('features', 1))

        # Get common columns and max rows from the selected files
        common_columns = get_common_columns(selected_files)
        max_rows = get_max_rows(selected_files)

        # Initialize the base ProcessDataForm
        form = ProcessDataForm(feature_count=features, common_columns=common_columns, user=request.user)

        # Create the context for rendering the features fields
        context = {
            'form': form,
            'range': range(features),
        }
        
        # Determine which dataset form to use based on the selected dataset type
        dataset_form = request.POST.get('dataset_form')
        if dataset_form == 'ts':
            form = ProcessTimeSeriesForm()
            form_html = render_to_string('datasets/process_ts_form.html', {'form': form}, request=request)
        else:
            form_html = ''

        # Render the feature fields HTML
        feature_fields_html = render_to_string('partials/dataset_feature_fields.html', context)

        # Return the response as a JSON object
        return JsonResponse({
            'columns': common_columns,
            'max_rows': max_rows,
            'feature_fields_html': feature_fields_html,
            'process_dataset_form_html': form_html,
        })

# Function that handles the logic for the process data form
def handle_process_data_form(request):
    if request.method == 'POST':
        # Extract the necessary data from POST
        feature_count = int(request.POST.get('features', 1))
        selected_files = request.POST.getlist('files')
        common_columns = get_common_columns(selected_files)
        user = request.user
        
        # Reinitialize the form with the same arguments as in `update_process_data_form`
        form = ProcessDataForm(request.POST, feature_count=feature_count, common_columns=common_columns, user=user)
        
        if form.is_valid():
            # Extract choices made by the user in the form
            dataset_title, dataset_comment, dataset_form, dataset_type, file_ids, start_row, end_row, features, feature_eng_choices = fetch_process_data_form_choices(form) 

            if dataset_form == 'ts':
                # Handle timeseries form choices
                form = ProcessTimeSeriesForm(request.POST)
                aggregation_method = fetch_ts_form_choices(form)
            if dataset_form == 'tabular':
                aggregation_method = None # No aggreagation needed for tabular data

            # Call the helper function to process and save the dataset
            response = process_and_save_dataset(request, dataset_title, dataset_comment, dataset_form, dataset_type, file_ids, start_row, end_row, features, feature_eng_choices, aggregation_method)
            if response:
                return response
        else:
            print(f'Form not valid: {form.errors}')
    else:
        form = ProcessDataForm()

    return render(request, 'datasets/process_data_form.html', {'form': form})

# Function to handle dataset processing and saving for ProcessDataForm
def process_and_save_dataset(request, dataset_title, dataset_comment, dataset_form, dataset_type, file_ids, start_row, end_row, features, feature_eng_choices, aggregation_method):
    # Create a DataFrame from the provided files and parameters
    dataset = create_custom_dataset(file_ids, features, start_row, end_row, feature_eng_choices, aggregation_method)

    # Create a database table from the DataFrame
    db = create_custom_db(dataset_title, dataset)

    # Populate the table with the table
    dataset_saved = create_model_instances(dataset, db)

    # Saves metadata entry if db table is created
    if db:
        user = request.user
        save_dataset_metadata(user, dataset_title, dataset_comment, dataset_form, dataset_type)  
    
    # Renders dataset sample if data is saved
    if dataset_saved:
        data, features = fetch_sample_dataset(dataset_title, 50)
        return render(request, 'datasets/sample_dataset.html', {'title': dataset_title, 'data': data, 'features': features})
    else:
        print(f'Dataset not saved: {dataset_title}')
        return None


####################################################################

# Function that handles the select model type form update
def update_build_model_form(request):
    build_model_form = request.POST.get('build_model_form')

    if build_model_form == 'sequential':
        seq_form = BuildSequentialForm(hidden_layer_count=1)
        form_html = render_to_string('models/build_models/build_sequential_model_form.html', {'seq_form': seq_form}, request=request)
    elif build_model_form == 'xgboost':
        form_html = render_to_string('models/build_models/build_xgboost_model_form.html', {}, request=request)
    elif build_model_form == 'mamba':
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
    feature_dataset_id = form.cleaned_data['feature_dataset']

    return model_title, model_comment, model_form, feature_dataset_id

# Function that handles the user choices for the sequential model form
def fetch_sequential_model_form_choices(form):
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

    return nodes, layer_types, activations, optimizer, loss, metrics

# Function that handles the logic for the build model form
def handle_build_model_form(request):
    if request.method == 'POST':
        form = BuildModelForm(request.POST)
        if form.is_valid():
            model_title, model_comment, model_form, feature_dataset_id = fetch_build_model_form_choices(form)
            
            if model_form == 'sequential':
                hidden_layer_count = int(request.POST.get('hidden_layers', 1))
                seq_form = BuildSequentialForm(request.POST, hidden_layer_count=hidden_layer_count)
                
                if seq_form.is_valid():
                    nodes, layer_types, activations, optimizer, loss, metrics = fetch_sequential_model_form_choices(seq_form)
                    user = request.user

                    model, model_id = build_sequential_model(model_title, user, model_comment, feature_dataset_id, layer_types, nodes, activations, optimizer, loss, metrics)
                    if model:
                        model_metadata = ModelMetadata.objects.get(id=model_id)
                        return redirect(reverse('view_model', kwargs={'model_id': model_metadata.id}))
                
                # If sequential form is invalid
                return render(request, 'models/build_models/build_model_form.html', {'form': form, 'form_errors': seq_form.errors})
            else:
                print(f'Model form not supported: {model_form}')
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

    feature_metadata = get_object_or_404(DatasetMetadata, id=feature_dataset_id)
    training_metadata = get_object_or_404(DatasetMetadata, id=training_dataset_id)
    model_metadata = get_object_or_404(ModelMetadata, id=model_id)

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

            feature_metadata = get_object_or_404(DatasetMetadata, id=feature_dataset_id)
            training_metadata = get_object_or_404(DatasetMetadata, id=training_dataset_id)

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
            model_metadata = ModelMetadata.objects.get(id=model_id)
            if history is not None:
                version_float = get_max_version(model_metadata.title)


            # Saves model and history as a .png
            user = request.user
            save_sequential_model(model_metadata.title, model, history, feature_dataset_id, 'sequential', version_float, user, model_metadata.comment)
            fig_url = plot_metrics(model_metadata.title, model_metadata.version, history.history)

            # Render the evaluation page with the plot
            return render(request, 'models/evaluate_model.html', {'fig_url': fig_url})
    else:
        form = TrainModelForm()

    return render(request, 'models/train_models/train_model_form.html', {'form': form})

#########################################################################

# Function that updates the prediction feature fields depending on which models are selected
def update_make_prediction_form(request):
    if request.method == 'POST':
        model_id = request.POST.get('model_id')
        if model_id:
            # Reinitialize the form with the selected model's features
            form = MakePredictionForm(user=request.user, model_id=model_id)
            model_features = load_features(model_id)
            features = model_features.keys()
            # Create the context for rendering the features fields
            context = {
                'form': form,
                'features': features,
            }

            features_fields_html = render_to_string(
                'partials/prediction_feature_fields.html',
                context,
            )

            return JsonResponse({'features_fields_html': features_fields_html})
        else:
            return JsonResponse({'error': 'Model ID not provided'}, status=400)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=400)

# Function to handle the logic for the make prediction form
def handle_make_prediction_form(request):
    user = request.user
    if request.method == 'POST':
        model_id = request.POST.get('model')
        form = MakePredictionForm(request.POST, user=user, model_id=model_id)
        if form.is_valid():
            # Loads the model object
            model_id = form.cleaned_data['model']
            model = prepare_model(model_id)

            features = prepare_features(form)

            prediction = model.predict(features)
            prediction_value = prediction[0]
            print(f'Prediction: {prediction_value}')
            context = {
                'form': form,
                'prediction': prediction_value
            }
            return render(request, 'models/make_prediction_form.html', context)
    else:
        form = MakePredictionForm(user=user)

    return render(request, 'models/make_prediction_form.html', {'form': form})