from __future__ import absolute_import, unicode_literals
import os
import re
import pandas as pd
from celery import shared_task
from django.core.exceptions import ValidationError
from django.conf import settings
from django.db import transaction, DatabaseError
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from .db_functions import merge_datetime

# Function that creates a custom dataset as a dataframe
def create_custom_dataset(file_paths, features, start_row, end_row, feature_eng_choices, aggregation_method):
    # Initialise an empty list to hold dataframes
    dataframes = []

    # Loop through each file and read it as DataFrame
    for file_path in file_paths:
        full_path = os.path.join(settings.USER_ROOT, file_path)
        chunk_iter = pd.read_csv(full_path, usecols=features, chunksize=100000)
        
        for chunk in chunk_iter:
            dataframes.append(chunk)

    # Concatenate all the DataFrames
    concat_df = pd.concat(dataframes, ignore_index=True)

    # Trim the DataFrame to the specified rows
    df = concat_df.iloc[start_row:end_row]

    if len(features) != len(feature_eng_choices):
        raise ValueError('Frequency of features and feature engineering choices do not match.')

    # Handle invalid column names
    cleaned_columns = clean_column_names(df.columns)
    df.columns = cleaned_columns

    # Identify date columns and create a subset DataFrame for date merging
    date_cols = [features[i] for i in range(len(feature_eng_choices)) if 'date_col' in feature_eng_choices[i]]
    if date_cols:
        date_df = df[date_cols]
        merged_date_df = merge_datetime(date_df, date_cols)
        df['datetime'] = merged_date_df['datetime']
        # Remove original date columns
        df.drop(columns=date_cols, inplace=True)
        print(f'Merged datetime columns: {date_cols}')

    # Create a list of remaining features after removing date columns
    remaining_features = [feature for feature in cleaned_columns if feature not in date_cols]
    remaining_choices = [choices for feature, choices in zip(features, feature_eng_choices) if feature not in date_cols]

    # Iterate over each column with its corresponding feature choices
    for feature, choices in zip(remaining_features, remaining_choices):
        # Use a temporary DataFrame slice to avoid SettingWithCopyWarning
        column_data = df[feature].copy()

        if 'handle_missing' in choices:
            column_data = handle_missing(column_data)
            df.loc[:, feature] = column_data
            print('Missing values handled.')

        if 'normalize' in choices:
            scaler = MinMaxScaler()
            column_data = scaler.fit_transform(column_data.values.reshape(-1, 1))  # Reshape to 2D
            df.loc[:, feature] = column_data.flatten().astype('float64')  # Flatten back to 1D if needed
            print('Dataset normalized.')

        if 'standardize' in choices:
            scaler = StandardScaler()
            column_data = scaler.fit_transform(column_data.values.reshape(-1, 1))  # Reshape to 2D
            df.loc[:, feature] = column_data.flatten().astype('float64')  # Flatten back to 1D if needed
            print('Dataset standardized')
    
    # Handle any aggregation method
    if aggregation_method:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.groupby(pd.Grouper(key='datetime', freq=aggregation_method)).mean().reset_index()
        df = df.dropna(subset=df.columns.difference(['datetime']))
        print(f'Data aggregated according to resolution: {aggregation_method}')

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


# Function to clean data by replacing erroneous values with zero
def handle_missing(df):
    # Replace -9999 with zero
    df.replace(-9999, 0, inplace=True)
    # Convert all values to floats and set non-numeric values to NaN
    df.apply(pd.to_numeric, errors='coerce')
    # Replace missing values with zero
    df.fillna(0, inplace=True)

    return df

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