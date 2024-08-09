import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from django.conf import settings
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU
from tensorflow.keras.optimizers import get as get_optimizer
from .db_functions import save_metadata

# Function that builds a keras neural network
def build_sequential_model(title, user, comment, layer_types, input_shape, nodes, activations, optimizer, loss, metrics):
    # Defines our model type
    model = Sequential()
    
    # Adjsut input shape for LSTM and GRU layers
    if layer_types[0] in ['LSTM', 'GRU']:
        input_shape = (None, input_shape[0]) # Time step dimension is None
    else:
        input_shape = input_shape # For dense layers, keep it as it

    # Add an input layer
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
    history = None
    save_sequential_model(title, model, history, 'sequential', 'untrained', user, comment)
    return model

# Function that saves a keras model based on if its trained or untrained
def save_sequential_model(title, model, history, model_form, trained_status, user, comment):
    # Determine the directory based on the training status
    if trained_status == 'trained':
        save_dir = os.path.join(settings.BASE_DIR, 'nn_models', 'trained', title)
    elif trained_status == 'untrained':
        save_dir = os.path.join(settings.BASE_DIR, 'nn_models', 'untrained', title)
    else:
        raise ValueError(f'Training status not recognized: {trained_status}')

    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Construct the file path
    model_path = os.path.join(save_dir, f'{title}.keras')

    # Save the model
    model.save(model_path)
    print(f'Model saved successfully to: {model_path}')

    # Save the history in JSON format
    if history is not None:
        history_path = os.path.join(save_dir, f'history_{title}.json')
        with open(history_path, 'w') as f:
            json.dump(history.history, f)

    # Saves the metadata
    save_metadata(title, comment, user, model_path, model_form, trained_status)

# Function which loads the training history of a model
def load_training_history(model_title):
    history_path = os.path.join(settings.BASE_DIR, 'nn_models', 'trained', model_title, f'history_{model_title}.json')
    with open(history_path, 'r') as f:
        history = json.load(f)
    return history

# Plotting function for a models metrics
def plot_metrics(model_title, history):
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
    fig_dir = os.path.join(settings.FIGURES_ROOT, model_title)
    os.makedirs(fig_dir, exist_ok=True)

    # Path to the figure file
    fig_path = os.path.join(fig_dir, 'metrics.png')
    fig_url = os.path.join(settings.FIGURES_URL, model_title, 'metrics.png')

    # Save the figure
    fig.savefig(fig_path)
    plt.close(fig)  # Close the figure to free memory

    return fig_url