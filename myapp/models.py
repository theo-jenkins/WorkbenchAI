from django.db import models, connection, DatabaseError
from django.apps import apps
from django.contrib.auth.models import AbstractUser
from django.conf import settings
from .form_choices import ACCOUNT_TYPE_CHOICES
import uuid
import pandas as pd

class CustomUser(AbstractUser):
    user_id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    first_name = models.CharField(max_length=30)
    email = models.EmailField(max_length=255, unique=True)
    username = None # Removes the username field
    type = models.CharField(max_length=50, choices=ACCOUNT_TYPE_CHOICES, default='developer')

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['first_name', 'type']

    def __str__(self):
        return self.email

class FileMetadata(models.Model):
    filename = models.CharField(max_length=255, unique=True)
    unique_id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    file_path = models.CharField(max_length=255)
    format = models.CharField(max_length=50) # 'csv', 'zip'

    class Meta:
        ordering = ['-uploaded_at']

    def __str__(self):
        return self.filename

class DatasetMetadata(models.Model):
    title = models.CharField(max_length=255)
    comment = models.TextField(blank=True, null=True)
    unique_id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    form = models.CharField(max_length=50) # 'tabular', 'ts'
    tag = models.CharField(max_length=50) # 'features', 'outputs'

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return self.title
    
class ModelMetadata(models.Model):
    title = models.CharField(max_length=255)
    comment = models.TextField(blank=True, null=True)
    unique_id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    file_path = models.CharField(max_length=255)
    tag = models.CharField(max_length=50) # 'untrained', 'trained'
    form = models.CharField(max_length=50) # 'sequential', 'xgboost', 'mamba'
    version = models.DecimalField( max_digits=5, decimal_places=2) # >1.0 indicates a trained model

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return self.title
    
def create_custom_db(title, dataset):
    try:
        class Meta:
            app_label = 'myapp'

        columns = dataset.columns

        attrs = {'__module__': 'myapp.models', 'Meta': Meta}
        for column in columns:
            value = dataset[column].iloc[0]  # Get the first value from the column
            
            # Check if the column's value is of numeric type
            if pd.api.types.is_numeric_dtype(value):
                attrs[column] = models.FloatField()  # Assign FloatField to the column
            
            # If the column's value is not numeric, treat it as a string
            else:
                attrs[column] = models.CharField(max_length=255)  # Assign CharField with max length of 255 to the column
        # Create the dynamic model
        model = type(title, (models.Model,), attrs)

        # Checks if the model is registered
        try:
            apps.get_model('myapp', title)
        except LookupError:
            apps.register_model('myapp', model)


        # Create the table in the database
        with connection.schema_editor() as schema_editor:
            schema_editor.create_model(model)
        
        return model
    except DatabaseError as e:
        print(f'An error occured creating the table: {e}')
        return None
    except Exception as e:
        print(f'An unexpected error occured: {e}')
        return None