import sqlite3
import pandas as pd
from django.db import models, connection
from django.conf import settings
from django.utils import timezone
from .models import Metadata

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

# Function that saves the metadata of a dataset or model
def save_metadata(title, comment, user, file_path, form, tag):
    # Django automatically generates a unique ID
    metadata = Metadata.objects.create(
        title=title,
        comment=comment,
        user=user,
        file_path=file_path,
        created_at=timezone.now(),
        form=form,
        tag=tag,
    )
    return metadata

# Function to retrieve a table from the db and return as df
def load_sqlite_table(db_path, table_name):
    conn = sqlite3.connect(db_path)
    query = f'SELECT * FROM "myapp_{table_name}"'
    df = pd.read_sql_query(query, conn)
    # Drop the first column (ID column)
    df = df.iloc[:, 1:]
    conn.close()
    return df

# Function that calculates the input shape of a selected dataset
def calc_dataset_shape(dataset_id):
    try:
        dataset_metadata = Metadata.objects.get(id=dataset_id)
    except Metadata.DoesNotExist:
        return None
    
    try:
        dataset = load_sqlite_table(dataset_metadata.file_path, dataset_metadata.title)
        return dataset.shape
    except Exception as e:
        print(f'Error loading SQLite tables: {e}')

# Function that merges date columns into a single datetime object column
def merge_datetime(df, date_cols):
    if len(date_cols) == 2: # Indicates date and time column
        date_col, time_col = date_cols
        df['datetime'] = pd.to_datetime(df[date_col] + ' ' + df[time_col], format='%d/%m/%Y %H:%M')
    elif len(date_cols) == 1: # Indicates just a date column
        date_col = date_cols[0]
        df['datetime'] = pd.to_datetime(df[date_col], format='%d/%m/%Y')
    return df
