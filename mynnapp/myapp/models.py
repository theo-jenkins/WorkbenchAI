from django.db import models, connection, DatabaseError
from django.apps import apps
from django.contrib.auth.models import AbstractUser
import uuid

class CustomUser(AbstractUser):
    user_id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    first_name = models.CharField(max_length=30)
    TYPE_CHOICES = [
        ('farmer', 'Farmer'),
        ('developer', 'Developer'),
    ]
    type = models.CharField(max_length=10, choices=TYPE_CHOICES, default='farmer')

    def __str__(self):
        return self.username

class UploadFile(models.Model):
    file = models.FileField(upload_to='myapp/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

def create_custom_db(title, columns):
    try:
        class Meta:
            app_label = 'myapp'

        attrs = {'__module__': 'myapp.models', 'Meta': Meta}
        for column in columns:
            attrs[column] = models.IntegerField()

        # Create the dynamic model
        model = type(title, (models.Model,), attrs)

        # Checks if the model is registered
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