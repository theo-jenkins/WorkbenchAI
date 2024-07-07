from django import forms
from django.core.exceptions import ValidationError
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm, UserChangeForm
from .models import CustomUser

class CustomUserCreationForm(UserCreationForm):
    class Meta(UserCreationForm):
        model = CustomUser
        fields = ('username', 'password1', 'password2', 'first_name', 'type')

    def save(self, commit=True):
        user = super(CustomUserCreationForm, self).save(commit=False)
        if commit:
            user.save()
        return user

class CustomUserChangeForm(UserChangeForm):
    class Meta:
        model = CustomUser
        fields = ('username', 'email', 'type')


def validate_file_extensions(file):
    valid_extensions = ['.csv']
    if not any(file.name.endswith(ext) for ext in valid_extensions):
        raise ValidationError('Invalid file extension.')
    
class UploadFileForm(forms.Form):
    file = forms.FileField(validators=[validate_file_extensions])

class ProcessDataForm(forms.Form):
    columns = forms.MultipleChoiceField(
        widget=forms.CheckboxSelectMultiple,
        required=True
        )
    start_row = forms.IntegerField(min_value=0, required=True)
    end_row = forms.IntegerField(min_value=0, required=True)