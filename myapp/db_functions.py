import sqlite3
import pandas as pd
import numpy as np
from django.db import models, connection
from django.conf import settings
from django.utils import timezone
from .models import FileMetadata, DatasetMetadata, ModelMetadata

# Function that saves the file metadata
def save_file_metadata(user, filename, file_path, format):
    metadata = FileMetadata.objects.create(
        filename=filename,
        file_path=file_path,
        user=user,
        format=format,
        uploaded_at=timezone.now(),
    )
    return metadata

# Function that saves the dataset metadata
def save_dataset_metadata(user, title, comment, form, tag):
    # Django automatically generates a unique ID
    metadata = DatasetMetadata.objects.create(
        title=title,
        comment=comment,
        user=user,
        created_at=timezone.now(),
        form=form,
        tag=tag,
    )
    return metadata

# Function that saves the model metadata
def save_model_metadata(user, title, comment, file_path, tag, form, version):
    # Django automatically generates a unique ID
    metadata = ModelMetadata.objects.create(
        title=title,
        comment=comment,
        user=user,
        created_at=timezone.now(),
        file_path=file_path,
        tag=tag,
        form=form,
        version=version,
    )
    return metadata

# Function that gets the columns from the db of type float
def get_float_columns(db):
    float_columns = []
    for field in db._meta.get_fields():
        if isinstance(field, models.FloatField):
            float_columns.append(field.name)
    return float_columns

# Function that returns the file_path of the sqlite db file
def get_db_file_path():
    db_path = settings.DATABASES['default']['NAME']
    return db_path

# Function that fetches a sample of the chosen dataset
def fetch_sample_dataset(title, sample_size):
    try:
        with connection.cursor() as cursor:
            # Ensure proper table name formatting to prevent SQL injection
            table_name = f'myapp_{title}'
            cursor.execute(f'SELECT * FROM "{table_name}" LIMIT %s', [sample_size])
            rows = cursor.fetchmany(sample_size)  # Fetch up to sample_size rows

            # Fetch column names
            columns = [col[0] for col in cursor.description]

            # Convert rows to list of dictionaries
            data = [dict(zip(columns, row)) for row in rows]

            return data, columns
    except Exception as e:
        print(f'An error occurred: {e}')
        return None, None

# Function to retrieve a table from the db and return as df
def load_sqlite_table(db_path, table_name, fetch='all', start_row=None, end_row=None, return_headers=False): 
    """
    Load data from an SQLite table with options for fetching all rows, a subset of rows, or just the headers.

    Args:
    - db_path: str, path to the SQLite database file.
    - table_name: str, name of the table to load data from.
    - fetch: str, options are 'all', 'range', 'headers'. Determines what data to fetch.
    - start_row: int, starting row for fetching a subset (optional, used with 'range').
    - end_row: int, ending row for fetching a subset (optional, used with 'range').
    - return_headers: bool, if True, only the headers (column names) are returned.

    Returns:
    - pd.DataFrame or list: DataFrame of the table data, or a list of column names if return_headers is True.
    """
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)

    # Create a SQL query based on the fetch type
    if fetch == 'headers' and return_headers:
        query = f'PRAGMA table_info("myapp_{table_name}")'
        headers = pd.read_sql(query, conn)['name'].tolist()
        conn.close()
        return headers
    
    elif fetch == 'range' and start_row is not None and end_row is not None:
        query = f'SELECT * FROM "myapp_{table_name}" LIMIT {end_row - start_row} OFFSET {start_row}'
    else:
        query = f'Select * FROM "myapp_{table_name}"'

    # Load the data into a DataFrame
    df = pd.read_sql_query(query, conn)
    # Drop the first column (ID column)
    df = df.iloc[:, 1:]
    conn.close()
    return df

# Function that returns the input shape of a dataset for training a keras model
def get_input_shape(dataset_id):
    try:
        dataset_metadata = DatasetMetadata.objects.get(id=dataset_id)
    except DatasetMetadata.DoesNotExist:
        return None
    try:
        db_file_path = get_db_file_path()
        dataset = load_sqlite_table(db_file_path, dataset_metadata.title, fetch='all')
    except Exception as e:
        print(f'Error loading SQLite tables: {e}')

    if dataset_metadata.form == 'tabular':
        num_features = dataset.shape[1]
        input_shape = (num_features,) # 1D input for Dense layers
    elif dataset_metadata.form == 'ts':
        num_features = dataset.shape[1] - 1 # Subtracting 1 for the datetime column
        input_shape = (None, num_features) # 2D input for LSTM and GRU layers
    else:
        input_shape = None

    return input_shape

# Function that merges date columns into a single datetime object column
def merge_datetime(df, date_cols):
    if len(date_cols) == 2: # Indicates date and time column
        date_col, time_col = date_cols
        df['datetime'] = pd.to_datetime(df[date_col] + ' ' + df[time_col], format='%d/%m/%Y %H:%M')
    elif len(date_cols) == 1: # Indicates just a date column
        date_col = date_cols[0]
        df['datetime'] = pd.to_datetime(df[date_col] + ' 00:00', format='%d/%m/%Y %H:%M')
    return df

# Function to prepare selected datasets for training
def prepare_datasets(features_id, outputs_id, timesteps):
    # Fetches the dataset metadata
    try:
        features_metadata = DatasetMetadata.objects.get(id=features_id)
        outputs_metadata = DatasetMetadata.objects.get(id=outputs_id)
    except DatasetMetadata.DoesNotExist:
        print(f'Dataset metadata could not be found: {features_id}, {outputs_id}')
        return None
    
    # Load features and outputs from SQLite database
    features, outputs = None, None
    db_file_path = get_db_file_path()
    try:
        features = load_sqlite_table(db_file_path, features_metadata.title, fetch='all')
        outputs = load_sqlite_table(db_file_path, outputs_metadata.title, fetch='all')
    except Exception as e:
        print(f'Error loading SQLite tables: {e}')

    if features is not None and not features.empty:
        print(f'Features loaded: {features_metadata.title}')
    else:
        print('Features data is empty or could not be loaded.')
        return None

    if outputs is not None and not outputs.empty:
        print(f'Outputs loaded: {outputs_metadata.title}')
    else:
        print('Outputs data is empty or could not be loaded.')
        return None

    # Check if both datasets are marked as time series ('ts')
    if features_metadata.form == 'ts' and outputs_metadata.form == 'ts':
        print('Both datasets are marked as timeseries. Aligning...')
        features, outputs = align_timeseries(features, outputs)
        features = reshape_features(features, timesteps)

    return features, outputs

# Function that aligns two timeseries dataframes to have the same datatime entries
def align_timeseries(df1, df2):
    # Convert the 'datetime' columns to datetime objects if they aren't already
    df1['datetime'] = pd.to_datetime(df1['datetime'])
    df2['datetime'] = pd.to_datetime(df2['datetime'])

    # Sort the dataframes by 'datetime'
    df1.sort_values(by='datetime', inplace=True)
    df2.sort_values(by='datetime', inplace=True)

    # Set the 'datetime' columns as the index for alignment
    df1.set_index('datetime', inplace=True)
    df2.set_index('datetime', inplace=True)

    # Create a common datetime index that both dataframes will be aligned to
    common_index = df1.index.intersection(df2.index)

    if common_index.empty:
        print('No common datetime index was found.')
    else:
        print('Common datatime index found.')        

    # Reindex both dataframes to the common datetime index
    aligned_df1 = df1.reindex(common_index)
    aligned_df2 = df2.reindex(common_index)

    # Drop the 'datetime' index (reset index without keeping it as a column)
    aligned_df1.reset_index(drop=True, inplace=True)
    aligned_df2.reset_index(drop=True, inplace=True)

    return aligned_df1, aligned_df2

# Function to reshape timeseries datasets
def reshape_features(features, timesteps):
    # Convert DataFrame to NumPy array
    data = features.values
    
    # Initialize empty list to collect sequences
    sequences = []
    
    # Create sequences of the specified length (timesteps)
    for i in range(len(data) - timesteps + 1):
        sequences.append(data[i:i + timesteps])
    
    # Convert the list of sequences to a NumPy array
    sequences = np.array(sequences)
    
    # The shape of the resulting array will be (num_samples, timesteps, num_features)
    return sequences