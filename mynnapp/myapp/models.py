from django.db import models
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