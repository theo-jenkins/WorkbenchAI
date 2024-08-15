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
2. **Upload a file** via the upload datasets form. Multiple file uploads are supported in .zip or .csv formats. Modifications to form validation such as validate_file_extensions and validate_file_size can be changed in forms.py
4. **Create your dataset** via the process datasets form. Choose from your uploaded files to select your features to be included. Mark the dataset form ('tabular' or 'timeseries') and dataset type ('features' or 'targets'). Process your features with the relevant feature engineering options.
5. **Build your neural network model** via the build model form. Choose your model type ('Sequential'). Specify each layers nodes, layer type and activation function. Choose your models optimizer, loss function and accuracy metrics.
6. **Train your neural network model** via the train model form. Choose your features, targets and model. Choose your training parameters ('batch size', 'epochs', 'verbose', 'validation split').
7. **View and evaluate** your models performance metrics.

## Project Structure

-myapp # Django app containing the core functionality
  -templates # HTML templates for the front-end
    -authentication
    -datasets
    -models
    -partials
    -predictions
  -apps.py
  -db_functions.py
  -form_choices.py 
  -forms.py
  -model_functions.py
  -models.py
  -site_functions.py
  -tasks.py
  -urls.py
  -views.py
-myproject # Django project settings
-figures # Directory for model performance figures
-nn_models # Directory for user created models
-uploaded_files # Directory for user uploaded files
-db.sqlite3 # Database
-requirements.txt # Python dependencies

## Technologies Used
- **Django**: Web framework for building the platform
- **Tensorflow/Keras**: Used for building and training neural networks
- **JavaScript/jQuery**: For dynamic form rendering and detecting client side interations
- **HTML**: For front-end development
- **SQLite**: Default database for development
