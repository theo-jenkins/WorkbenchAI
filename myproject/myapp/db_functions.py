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
            cursor.execute(f'SELECT * FROM {table_name} LIMIT %s', [sample_size])
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
def save_metadata(title, comment, user, file_path, tag):
    # Django automatically generates a unique ID
    metadata = Metadata.objects.create(
        title=title,
        comment=comment,
        user=user,
        file_path=file_path,
        created_at=timezone.now(),
        tag=tag,
    )
    return metadata