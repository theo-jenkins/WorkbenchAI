import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from django.conf import settings
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, GRU
from tensorflow.keras.optimizers import get as get_optimizer
from .db_functions import save_model_metadata, get_db_file_path, load_sqlite_table, get_input_shape
from .models import DatasetMetadata, ModelMetadata

# Function that builds a keras neural network
def build_sequential_model(title, user, comment, feature_dataset_id, layer_types, nodes, activations, optimizer, loss, metrics):
    # Defines our model type
    model = Sequential()
    
    # Add an input layer
    input_shape = get_input_shape(feature_dataset_id)
    model.add(Input(shape=input_shape))

    # Correctly initializing the input layer
    if layer_types[0] == 'dense':
        # Assuming nodes[0] is the size of the input feature set
        model.add(Dense(nodes[0], activation=activations[0]))
    elif layer_types[0] == 'LSTM':
        model.add(LSTM(nodes[0], activation=activations[0], return_sequences=(len(layer_types) > 1)))
    elif layer_types[0] == 'GRU':
        model.add(GRU(nodes[0], activation=activations[0], return_sequences=(len(layer_types) > 1)))

    # Adding subsequent layers
    for i in range(1, len(layer_types)):  # Start from 1 since 0 is already added
        if layer_types[i] == 'dense':
            model.add(Dense(nodes[i], activation=activations[i]))
        elif layer_types[i] == 'LSTM':
            model.add(LSTM(nodes[i], activation=activations[i], return_sequences=(i < len(layer_types) - 1)))
        elif layer_types[i] == 'GRU':
            model.add(GRU(nodes[i], activation=activations[i], return_sequences=(i < len(layer_types) - 1)))

    # Compile the model
    model.compile(optimizer=get_optimizer(optimizer), loss=loss, metrics=metrics)

    # Saves the model as a file
    model_id = save_sequential_model(title, model, None, feature_dataset_id, 'sequential', 0.0, user, comment)
    return model, model_id

# Function that saves a keras model based on if its trained or untrained
def save_sequential_model(title, model, history, feature_dataset_id, model_form, version, user, comment):
    # Determine the base directory based on the model title
    base_dir = os.path.join(settings.MODEL_ROOT, title)
    print(f'Version string: {version}')
    version_dir = os.path.join(base_dir, f'Version_{version}')
    
    # Create the necessary directories if they don't exist
    os.makedirs(version_dir, exist_ok=True)

    # Construct the file paths
    model_path = os.path.join(version_dir, f'{title}.keras')
    history_path = os.path.join(version_dir, f'{title}_history.json')

    # Save the model
    try:
        model.save(model_path)
        print(f'Model saved successfully to: {model_path}')
    except Exception as e:
        print(f'Error saving model: {e}')

    # Save the history in JSON format, if it exists
    if history:
        try:
            with open(history_path, 'w') as f:
                json.dump(history.history, f)
            print(f'History saved successfully to: {history_path}')
        except Exception as e:
            print(f'Error saving history: {e}')

    # Save the features in JSON format
    try:
        save_features(title, feature_dataset_id)
        print(f'Features saved for model: {title}')
    except Exception as e:
        print(f'Error saving features: {e}')

    # Determine the trained status based on the version
    trained_status = 'trained' if version != 0.0 else 'untrained'

    # Save metadata
    try:
        model_metadata = save_model_metadata(user, title, comment, model_path, trained_status, model_form, version)
        print(f'Metadata saved for model: {title}')
    except Exception as e:
        print(f'Error saving metadata: {e}')
    
    model_id = model_metadata.id
    return model_id

# Functions that saves the feature names for a model in json format
def save_features(model_title, dataset_id):
    db_file_path = get_db_file_path()
    database_metadata = DatasetMetadata.objects.get(id=dataset_id)
    dataset_headers = load_sqlite_table(db_file_path, database_metadata.title, return_headers=True)
     # Convert the list of dataset headers to a dictionary where keys are the column names
    features_dict = {column: None for column in dataset_headers}
    
    json_file = os.path.join(settings.MODEL_ROOT, model_title, f'{model_title}_features.json')
    
    with open(json_file, 'w') as f:
        json.dump(features_dict, f)

    return json_file

# Function that loads the features of a dataset and returns a dictionary
def load_features(model_id):
    model_metadata = ModelMetadata.objects.get(id=model_id)
    json_file = os.path.join(settings.MODEL_ROOT, model_metadata.title, f'{model_metadata.title}_features.json')
    with open(json_file, 'r') as f:
        features = json.load(f)
    return features

# Function which loads the training history of a model
def load_training_history(model_title, model_version):
    history_path = os.path.join(settings.MODEL_ROOT, model_title, f'Version_{model_version}', f'{model_title}_history.json')
    with open(history_path, 'r') as f:
        history = json.load(f)
    return history

# Plotting function for a models metrics
def plot_metrics(model_title, model_version, history):
    metrics = [key for key in history.keys() if not key.startswith('val_')]
    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 5))

    if num_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        ax.plot(history[metric], label='Train')
        val_metric = f'val_{metric}'
        if val_metric in history:
            ax.plot(history[val_metric], label='Validation')
        ax.set_title(f'Model {metric}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.legend(loc='upper left')

    # Ensure the directory exists
    fig_dir = os.path.join(settings.FIGURES_ROOT, model_title, f'Version_{model_version}')
    os.makedirs(fig_dir, exist_ok=True)

    # Path to the figure file
    fig_path = os.path.join(fig_dir, 'metrics.png')
    fig_url = os.path.join(settings.FIGURES_URL, model_title, f'Version_{model_version}', 'metrics.png')

    # Save the figure
    fig.savefig(fig_path)
    plt.close(fig)  # Close the figure to free memory

    return fig_url

# Function to fetch and load a keras model
def prepare_model(model_id):
    try:
        model_metadata = ModelMetadata.objects.get(id=model_id)
    except ModelMetadata.DoesNotExist:
        print(f'Model metadata could not be found: {model_id}')
        return None
    
    # Load keras model
    model = None
    try:
        model = load_model(model_metadata.file_path)
    except Exception as e:
        print(f'Error loading keras model: {e}')
    if model:
        print(f'Model loaded: {model_metadata.title}')
        
    return model

def fetch_gpu_info():
    # Check for GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        message = (f'GPU(s) detected.')
    else:
        message = 'No GPUs detected. The model will use the CPU for training.'

    return message

# Function that returns the version number of a model
def get_max_version(model_title):
    model_dir = os.path.join(settings.MODEL_ROOT, model_title)
    # Check if model has been trained
    if os.path.exists(model_dir):
        untrained_dir = os.path.join(model_dir, f'Version_0.0')
        if os.path.exists(untrained_dir):
            version = 1.0 # Directory already exists, increment version number
        trained_dir = os.path.join(model_dir, f'Version_1.0')
        while os.path.exists(trained_dir):
            version += 0.1 # Directory already exists, increment version number
            trained_dir = os.path.join(model_dir, f'Version_{version}')
        return version
    else:
        print('Model could not be found, version could not be determined.')
        return False

# Function that prepares features for prediction
def prepare_features(form):
    model_id = form.cleaned_data['model']
    model_features = load_features(model_id)
    features = []
    for feature in model_features.keys():
        feature_value = form.cleaned_data.get(feature)  # Safely retrieve the value
        if feature_value is not None:
            features.append(float(feature_value))  # Convert to float if needed
        else:
            features.append(0.0)  # Default to 0.0 if the value is missing
    features  = np.array([features])  # Convert to a 2D array

    return features
