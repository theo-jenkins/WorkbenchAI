from django.db import models, connection, DatabaseError
from django.apps import apps
from django.contrib.auth.models import AbstractUser
from django.conf import settings
import uuid
import re
import pandas as pd

class CustomUser(AbstractUser):
    user_id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    first_name = models.CharField(max_length=30)
    email = models.EmailField(max_length=255, unique=True)
    TYPE_CHOICES = [
        ('farmer', 'Farmer'),
        ('developer', 'Developer'),
    ]
    type = models.CharField(max_length=50, choices=TYPE_CHOICES, default='developer')

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['first_name']

    def __str__(self):
        return self.email

class Metadata(models.Model):
    title = models.CharField(max_length=255)
    comment = models.TextField(blank=True, null=True)
    unique_id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    file_path = models.CharField(max_length=255)
    form = models.CharField(max_length=50) # 'tabular', 'ts', 'sequential', 'xgboost'
    tag = models.CharField(max_length=50) # 'features', 'outputs', 'untrained', 'trained'

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
            value = dataset[column].iloc[0]  # Get the first value from the columns
            
            # Check if the column's value matches hour time format (hh:mm)
            if isinstance(value, str) and re.match(r'^\d{2}:\d{2}$', value):
                attrs[column] = models.TimeField()  # Assign TimeField to the column
            
            # Check if the column's value matches date format (dd/mm/yyyy)
            elif isinstance(value, str) and re.match(r'^\d{2}/\d{2}/\d{4}$', value):
                attrs[column] = models.DateField()  # Assign DateField to the column
                        
            # Check if the column's value is of numeric type
            elif pd.api.types.is_numeric_dtype(value):
                attrs[column] = models.FloatField()  # Assign FloatField to the column
            
            # If the column's value is neither datetime, time, date nor numeric, treat it as a string
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