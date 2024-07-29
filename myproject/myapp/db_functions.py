from django.db import models
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

# Function that fetches the first 50 rows of a chosen database table
def fetch_sample_dataset(db, sample_size):
    db_data = db.objects.all().values()[:sample_size]  # Get the first 50 rows
    columns = db_data[0].keys() if db_data else []  # Get column names

    return db_data, columns

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