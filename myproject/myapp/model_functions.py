import os
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
    save_model(title, model, 'untrained', user, comment)
    return model

# Function that saves a keras model based on if its trained or untrained
def save_model(title, model, trained_status, user, comment):
    # Determine the directory based on the training status
    if trained_status == 'trained':
        save_dir = os.path.join(settings.BASE_DIR, 'nn_models', 'trained')
    elif trained_status == 'untrained':
        save_dir = os.path.join(settings.BASE_DIR, 'nn_models', 'untrained')
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

    # Saves the metadata
    save_metadata(title, comment, user, model_path, trained_status)