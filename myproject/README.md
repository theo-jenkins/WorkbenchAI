# My Django Web App

A powerful web application that allows users to build and train neural networks.

## Table of Contents:
- [Features](#features)
- [Installation](#installation)
- [Work Flow](#work-flow)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)

## Features
- Account creation
- User authentication and authorization
- File upload
- Process datasets
- Build neural network models
- Train neural network models
- View model performance metrics
- Dynamic form rendering

## Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/theo-jenkins/nn_sandbox.git
   cd your-repo-name
   ```

2. **Create a virtual environment and activate it**:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up the database**:
   ```bash
   python manage.py migrate
   ```

5. **Run the development server**:
   ```bash
   python manage.py runserver
   ```

## Work Flow
- Instructions on how to use the application.
1. **Create an account** or **log in**
    - To access all the features, create and login in to your account.
2. **Upload a file** via the upload datasets form.
    - You can upload one or multiple files in '.zip' or '.csv' formats.
    - If needed, you can adjust the form validation rules such as 'validate_file_extensions' and 'validation_file_size' found in the forms.py to match your requirements.
4. **Create your dataset** via the process datasets form.
    - Select from your uploaded files to define the features included in the dataset.
    - Specify the dataset form ('tabular' or 'timeseries') and dataset type ('features' or 'targets').
    - Apply relevant feature engineering options to preprocess your features.
5. **Build your neural network model** via the build model form.
    - Choose the model type (currently supported: 'Sequential').
    - Define the architecture by specifying each layer’s nodes, layer type, and activation function.
    - Select the optimizer, loss function, and accuracy metrics for your model.
6. **Train your neural network model** via the train model form.
    - Select the features, targets, and model you wish to train.
    - Configure your training parameters ('batch size', 'epochs', 'verbose', 'validation split').
    - Start the training process.
7. **View and evaluate** your models performance metrics.
    - After training, review the model’s performance metrics, including accuracy, loss, and visualizations like training curves.

## Project Structure

### myapp # Django app containing the core functionality
  - templates/ # HTML templates for the front-end
    - authentication/ # Templates related to user authentication
    - datasets/ # Templates for dataset management and processing
    - models/ # Templates for model building and evaluation
    - partials/ # Reusable template components (e.g., form partials)
    - predictions/ # Templates for making and viewing predictions
  - apps.py # Django app configuration
  - db_functions.py # Database-related utility functions
  - form_choices.py # Choices used in forms (e.g., dropdown options)
  - forms.py # Django forms used in the application
  - model_functions.py # Functions related to model creation and training
  - models.py # Django models representing the database schema
  - site_functions.py # General utility functions for the site
  - tasks.py # Background tasks and processing functions
  - urls.py # URL routing for the Django app
  - views.py # Views that handle the logic for each web page

### myproject # Django project settings
  - settings.py # Django project settings
  - urls.py # URL routing for the Django project
  - wsgi.py # WSGI configuration for deployment
  - asgi.py # ASGI configuration for asynchronous deployment

### figures/ # Directory for storing model performance figures (e.g., accuracy plots)
### nn_models/ # Directory for storing user-created models
### uploaded_files/ # Directory for storing user-uploaded files
### db.sqlite3 # SQLite database file for development
### requirements.txt # Python dependencies

## Technologies Used
- **Django**: Web framework for building the platform
- **Tensorflow/Keras**: Used for building and training neural networks
- **JavaScript/jQuery**: For dynamic form rendering and detecting client side interations
- **HTML**: For front-end development
- **SQLite**: Default database for development