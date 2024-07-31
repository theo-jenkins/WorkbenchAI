from __future__ import absolute_import, unicode_literals
import os
import re
import pandas as pd
import zipfile
from celery import shared_task
from django.core.exceptions import ValidationError
from django.conf import settings
from django.db import transaction, DatabaseError
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# Function that validates and saves the file and deletes the .zip if applicable
def upload_files(files):
    valid_extensions = ['.zip', '.csv']
    processed_files = []
    
    # Ensure the uploaded_files directory exists
    upload_dir = os.path.join(settings.MEDIA_ROOT)
    os.makedirs(upload_dir, exist_ok=True)
    
    for file in files:
        if not any(file.name.endswith(ext) for ext in valid_extensions):
            print(f'Invalid file extension: {file.name}')
            continue  # Skip to the next file

        try:
            file_path = os.path.join(upload_dir, file.name)
            # Save the uploaded file to the directory
            with open(file_path, 'wb+') as destination:
                for chunk in file.chunks():
                    destination.write(chunk)
            
            # Handle .zip files
            if file.name.endswith('.zip'):
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    for zip_info in zip_ref.infolist():
                        if zip_info.filename.endswith('.csv'):
                            zip_info.filename = os.path.basename(zip_info.filename)  # Remove any directory structure
                            zip_ref.extract(zip_info, upload_dir)
                os.remove(file_path)  # Remove the .zip file after extraction
            
            # Append successfully processed file
            processed_files.append(file.name)
        
        except Exception as e:
            print(f'An error occured uploading the file: {e}')
            continue  # Proceed with the next file

    return processed_files  # Return list of processed files
    

# Function that creates a custom dataset as a dataframe
def create_custom_dataset(file_paths, features, start_row, end_row, feature_eng_choices):
    # Initialise an empty list to hold dataframes
    dataframes = []

    # Loop through each file and read it as DataFrame
    for file_path in file_paths:
        full_path = os.path.join(settings.MEDIA_ROOT, file_path)
        chunk_iter = pd.read_csv(full_path, usecols=features, chunksize=100000)
        
        for chunk in chunk_iter:
            dataframes.append(chunk)

    # Concatenate all the DataFrames
    concat_df = pd.concat(dataframes, ignore_index=True)

    # Trim the DataFrame to the specified rows
    trimmed_df = concat_df.iloc[start_row:end_row]
    df = trimmed_df

    if len(features) != len(feature_eng_choices):
        raise ValueError('Frequency of features and feature engineering choices do not match.')

    # Handle invalid column names
    cleaned_columns = clean_column_names(df.columns)
    cleaned_features = clean_column_names(features)
    df.columns = cleaned_columns

    # Debuggings
    print(f'Original columns: {features}')
    print(f'New columns: {cleaned_columns}')

    # Iterate over each column with its corresponding feature choices
    for feature, choices in zip(cleaned_features, feature_eng_choices):
        # Use a temporary DataFrame slice to avoid SettingWithCopyWarning
        column_data = df[[feature]].copy()

        if 'handle_missing' in choices:
            df[feature] = clean_data(df[feature])
            print('Missing values handled.')

        if 'normalize' in choices:
            scaler = MinMaxScaler()
            column_data = scaler.fit_transform(column_data)
            df[feature] = column_data  # Update the original DataFrame
            print('Dataset normalized.')

        if 'standardize' in choices:
            scaler = StandardScaler()
            column_data = scaler.fit_transform(column_data)
            df[feature] = column_data  # Update the original DataFrame
            print('Dataset standardized')

    return df

# Function to reformat dd/mm/yyyy into yyyy-mm-dd datetime objects
def format_dates(df):
    date_cols = df.select_dtypes(include=['object']).columns
    for col in date_cols:
        # Check if the first entry matches the dd/mm/yyyy format
        if re.match(r'^\d{2}/\d{2}/\d{4}$', df[col].iloc[0]):
            # Convert entire column to datetime and reformat to yyyy-mm-dd
            df[col] = pd.to_datetime(df[col], format='%d/%m/%Y', errors='coerce').dt.strftime('%Y-%m-%d')
    
    return df

# Function to clean data by replacing erroneous values with zero
def clean_data(df):
    # Replace -9999 with zero
    df.replace(-9999, 0, inplace=True)
    # Handle invalid date formats
    #df = format_dates(df)
    # Convert all values to floats and set non-numeric values to NaN
    df = df.apply(pd.to_numeric, errors='coerce')
    # Replace missing values with zero
    df.fillna(0, inplace=True)
    return df

# Function to clean column names
def clean_column_names(columns):
    valid_columns = []
    for col in columns:
        # Replace invalid characters with underscores
        valid_col = re.sub(r'[^a-zA-Z0-9_]', '_', col)
        # Ensure the column name doesn't start with a digit
        if re.match(r'^[0-9]', valid_col):
            valid_col = '_' + valid_col
        valid_columns.append(valid_col)
    return valid_columns

# Function that converts a dataset into a list of model instances
def create_model_instances(dataset, db):
    try:
        # Initializes batch size for entries saved to db
        batch_size = 500
        # Convert DataFrame rows to a list of model instances with progress printing
        entries = []
        total_rows = len(dataset)
        for index, row in dataset.iterrows():
            entries.append(db(**row.to_dict()))

            # Progress printing
            if (index + 1) % batch_size == 0 or (index + 1) == total_rows:
                print(f'Entries converted: {index + 1}/{total_rows}')

        # Checks if all the columns of the dataset match the db
        db_columns = [field.name for field in db._meta.get_fields()]
        dataset_columns = dataset.columns
        if all(column in db_columns for column in dataset_columns):
            success = commit_to_db(entries, db, batch_size)
            return success
    except Exception as e:
        print(f'An error occured while creating model instances: {e}')
        return False

# Function that commits entries to db
def commit_to_db(entries, db, batch_size):
    try:
        # Save entries in batches
        with transaction.atomic():
            for i in range(0, len(entries), batch_size):
                db.objects.bulk_create(entries[i:i+batch_size])
                print(f'Entries committed: {i}/{len(entries)}.')
        return True
    except DatabaseError as e:
        print(f'An error occured while creating the table: {e}')
        return False
    except Exception as e:
        print(f'An unexpected error occured: {e}')
        return False


# Function that trains the selected model
def train_model(features, output, model, batch_size, epochs, verbose, validation_split):
    history = model.fit(features, output,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=verbose,
                    validation_split=validation_split)
    return history, model