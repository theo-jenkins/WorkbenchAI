import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from django.conf import settings
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from .db_functions import save_metadata
# Function that builds a keras neural network
def build_model(title, user, comment, layer_types, nodes, activations, optimizer, loss, metrics):
    model = Sequential()

    # Correctly initializing the input layer
    if layer_types[0] == 'dense':
        # Assuming nodes[0] is the size of the input feature set
        model.add(Dense(nodes[0], input_shape=(nodes[0],), activation=activations[0]))

    # Adding subsequent layers
    for i in range(1, len(layer_types)):  # Start from 1 since 0 is already added
        if layer_types[i] == 'dense':
            model.add(Dense(nodes[i], activation=activations[i]))

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Saves the model as a file
    history = None
    save_model(title, model, history, 'untrained', user, comment)
    return model

# Function that saves a keras model based on if its trained or untrained
def save_model(title, model, history, trained_status, user, comment):
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
    history_path = os.path.join(save_dir, f'history_{title}.json')
    with open(history_path, 'w') as f:
        json.dump(history.history, f)

    # Saves the metadata
    save_metadata(title, comment, user, model_path, trained_status)

# Function which loads the training history of a model
def load_training_history(model_title):
    history_path = os.path.join(settings.BASE_DIR, 'nn_models', 'trained', model_title, f'history_{model_title}.json')
    with open(history_path, 'r') as f:
        history = json.load(f)
    return history

# Plotting function for a models metrics
def plot_metrics(history, save_path):
    metrics = [key for key in history.keys() if not key.startswith('val_')]
    validation_metrics = [f'val_{metric}' for metric in metrics if f'val_{metric}' in history]

    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 5))

    if num_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        ax.plot(history[metric])
        if f'val_{metric}' in history:
            ax.plot(history[f'val_{metric}'])
        ax.set_title(f'Model {metric}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.legend(['Train', 'Validation'] if f'val_{metric}' in history else ['Train'], loc='upper left')

    # Save the figure
    fig.savefig(save_path)
    plt.close(fig)  # Close the figure to free memory

    return save_path