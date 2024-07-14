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
    valid_extensions = ['.zip', '.csv']
    if not any(file.name.endswith(ext) for ext in valid_extensions):
        raise ValidationError('Invalid file extension.')
    
def validate_file_size(file):
    max_size_mb = 100000 # Maximum file size in MB
    if file.size > max_size_mb * 1024 * 1024:
        raise ValidationError(f'File size can not exceed {max_size_mb} MB.')

class UploadFileInput(forms.ClearableFileInput):
    allow_multiple_selected = True

class UploadFileField(forms.FileField):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("widget", UploadFileInput())
        super().__init__(*args, **kwargs)

    def clean(self, data, initial=None):
        single_file_clean = super().clean
        if isinstance(data, (list, tuple)):
            result = [single_file_clean(d, initial) for d in data]
        else:
            result = [single_file_clean(data, initial)]
        return result

class UploadFileForm(forms.Form):
    file_field = UploadFileField(
        validators=[validate_file_extensions, validate_file_size],
    )

class ProcessDataForm(forms.Form):
    db_title = forms.CharField(
        required=True,
    )
    comment = forms.CharField(
        widget=forms.Textarea,
        required=False,
    )
    files = forms.MultipleChoiceField(
        widget=forms.CheckboxSelectMultiple,
        required=True,
    )
    columns = forms.MultipleChoiceField(
        widget=forms.CheckboxSelectMultiple,
        required=False,
    )
    feature_eng = forms.MultipleChoiceField(
        widget=forms.CheckboxSelectMultiple,
        required=False,
    )
    start_row = forms.IntegerField(
        initial=0,
    )
    end_row = forms.IntegerField(
        required=True,
    )

class BuildModelForm(forms.Form):
    model_title = forms.CharField(
        required=True,
    )
    comment = forms.CharField(
        widget = forms.Textarea,
        required=False,
    )
    model_type = forms.ChoiceField(
        required=True,
    )
    features = forms.IntegerField(
        required=True,
        initial=1,
    )
    hidden_layers = forms.IntegerField(
        required=True,
        initial=1,
    )
    outputs = forms.IntegerField(
        required=True,
        initial=1,
    )
    optimizer = forms.ChoiceField(
        required=True,
    )
    loss = forms.ChoiceField(
        required=True,
    )
    metrics = forms.MultipleChoiceField(
        widget=forms.CheckboxSelectMultiple,
        required=True,
    )

class TrainModelForm(forms.Form):
    chosen_dataset = forms.MultipleChoiceField(
        widget=forms.CheckboxSelectMultiple,
        required=True
    )
    chosen_model = forms.BooleanField(
        required=True,
    )