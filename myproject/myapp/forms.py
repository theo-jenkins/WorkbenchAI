from django import forms
from django.core.exceptions import ValidationError
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm, UserChangeForm
from .models import CustomUser, Metadata
from .site_functions import get_uploaded_files
from .form_choices import FEATURE_ENG_CHOICES, DATASET_FORM_CHOICES, DATASET_TYPE_CHOICES, AGGREGATION_FREQUENCY_CHOICES, MODEL_FORM_CHOICES, LAYER_TYPE_CHOICES, ACTIVATION_TYPE_CHOICES, OPTIMIZER_CHOICES, LOSS_CHOICES, METRIC_CHOICES

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
    dataset_form = forms.ChoiceField(
        required=True,
        choices=DATASET_FORM_CHOICES,
        widget=forms.RadioSelect,
    )
    dataset_type = forms.ChoiceField(
        required=True,
        choices=DATASET_TYPE_CHOICES,
        widget=forms.RadioSelect,
    )

class ProcessTabularForm(forms.Form):
    def __init__(self, *args, **kwargs):
        feature_count = kwargs.pop('feature_count', 1)
        common_columns = kwargs.pop('common_columns', [])
        super(ProcessTabularForm, self).__init__(*args, **kwargs)

        self.fields['files'].choices = self.get_file_choices()

        for i in range(feature_count):
            self.fields[f'column_{i}'] = forms.ChoiceField(
                required=True,
                label=f'Feature {i}',
                choices=[(col, col) for col in common_columns],
            )
            self.fields[f'feature_eng_{i}'] = forms.MultipleChoiceField(
                widget=forms.CheckboxSelectMultiple,
                required=False,
                choices=FEATURE_ENG_CHOICES,
                label=f'Feature Engineering Options {i}',
            )
    @staticmethod
    def get_file_choices():
        files = get_uploaded_files()
        return files
    
    files = forms.MultipleChoiceField(
        widget=forms.CheckboxSelectMultiple,
        required=True,
    )
    start_row = forms.IntegerField(
        initial=0,
    )
    end_row = forms.IntegerField(
        required=True,
    )
    features = forms.IntegerField(
        required=True,
        initial=1,
        min_value=1,
    )

class ProcessTimeSeriesForm(forms.Form):
    def __init__(self, *args, **kwargs):
        feature_count = kwargs.pop('feature_count', 1)
        common_columns = kwargs.pop('common_columns', [])
        super(ProcessTimeSeriesForm, self).__init__(*args, **kwargs)

        self.fields['files'].choices = self.get_file_choices()

        for i in range(feature_count):
            self.fields[f'column_{i}'] = forms.ChoiceField(
                required=True,
                label=f'Feature {i}',
                choices=[(col, col) for col in common_columns],
            )
            self.fields[f'feature_eng_{i}'] = forms.MultipleChoiceField(
                widget=forms.CheckboxSelectMultiple,
                required=False,
                choices=FEATURE_ENG_CHOICES,
                label=f'Feature Engineering Options {i}',
            )
    @staticmethod
    def get_file_choices():
        files = get_uploaded_files()
        return files
    
    aggregation_method = forms.ChoiceField(
        required=True,
        choices=AGGREGATION_FREQUENCY_CHOICES,
        widget=forms.RadioSelect,
    )
    files = forms.MultipleChoiceField(
        widget=forms.CheckboxSelectMultiple,
        required=True,
    )
    start_row = forms.IntegerField(
        initial=0,
    )
    end_row = forms.IntegerField(
        required=True,
    )
    features = forms.IntegerField(
        required=True,
        initial=1,
        min_value=1,
    )


class BuildModelForm(forms.Form):
    def __init__(self, *args, **kwargs):
        super(BuildModelForm, self).__init__(*args, **kwargs)

        # Query for datasets tagged as 'features' or 'outputs'
        feature_datasets = Metadata.objects.filter(tag='features')

        # Prepare choices as tuples (id, title)
        feature_choices = [(dataset.id, dataset.title) for dataset in feature_datasets]

        # Set the choices for the form fields
        self.fields['feature_dataset'].choices = feature_choices

    model_title = forms.CharField(
        required=True,
        label='Model Title',
    )
    comment = forms.CharField(
        widget=forms.Textarea(),
        required=False,
        label='Comment',
    )
    model_form = forms.ChoiceField(
        required=True,
        choices=MODEL_FORM_CHOICES,
        widget=forms.RadioSelect,
    )
    feature_dataset = forms.ChoiceField(
        required=True,
    )
    
class BuildSequentialForm(forms.Form):
    def __init__(self, *args, **kwargs):
        hidden_layer_count = kwargs.pop('hidden_layer_count', 1)
        super(BuildSequentialForm, self).__init__(*args, **kwargs)

        for i in range(hidden_layer_count):
            self.fields[f'nodes_{i}'] = forms.IntegerField(
                label=f'Nodes {i+1}',
                required=True,
                min_value=1,
                initial=32,
            )
            self.fields[f'layer_type_{i}'] = forms.ChoiceField(
                label=f'Hidden Layer Type {i+1}',
                choices=LAYER_TYPE_CHOICES,
                required=True,
            )
            self.fields[f'activation_{i}'] = forms.ChoiceField(
                label=f'Activation Type {i+1}',
                choices=ACTIVATION_TYPE_CHOICES,
                required=True,
            )
    input_nodes = forms.IntegerField(
        required=True,
        initial=1,
        min_value=1,
    )
    input_layer_type = forms.ChoiceField(
        choices=LAYER_TYPE_CHOICES,
        required=True,
    )
    input_activation = forms.ChoiceField(
        choices=ACTIVATION_TYPE_CHOICES,
        required=True,
    )
    outputs = forms.IntegerField(
        required=True,
        initial=1,
        min_value=1,
    )
    output_layer_type = forms.ChoiceField(
        choices=LAYER_TYPE_CHOICES,
        required=True,
    )
    output_activation = forms.ChoiceField(
        choices=ACTIVATION_TYPE_CHOICES,
        required=True,
    )
    hidden_layers = forms.IntegerField(
        required=True,
        initial=1,
        min_value=0,
    )    
    optimizer = forms.ChoiceField(
        choices=OPTIMIZER_CHOICES,
        required=True,
    )
    loss = forms.ChoiceField(
        choices=LOSS_CHOICES,
        required=True,
    )
    metrics = forms.MultipleChoiceField(
        widget=forms.CheckboxSelectMultiple,
        choices=METRIC_CHOICES,
        required=True,
    )
    
class TrainModelForm(forms.Form):
    def __init__(self, *args, **kwargs):
        super(TrainModelForm, self).__init__(*args, **kwargs)

        # Query for datasets tagged as 'features' or 'outputs'
        feature_datasets = Metadata.objects.filter(tag='features')
        training_datasets = Metadata.objects.filter(tag='outputs')
        
        # Query for models tagged as 'untrained'
        untrained_models = Metadata.objects.filter(tag='untrained')

        # Prepare choices as tuples (id, title)
        feature_choices = [(dataset.id, dataset.title) for dataset in feature_datasets]
        training_choices = [(dataset.id, dataset.title) for dataset in training_datasets]
        model_choices = [(model.id, model.title) for model in untrained_models]

        # Set the choices for the form fields
        self.fields['feature_dataset'].choices = feature_choices
        self.fields['training_dataset'].choices = training_choices
        self.fields['model'].choices = model_choices

    model_title = forms.CharField(
        required=True,
        label='Model Title',
    )
    comment = forms.CharField(
        widget=forms.Textarea(),
        required=False,
    )
    feature_dataset = forms.ChoiceField(
        required=True,
    )
    training_dataset = forms.ChoiceField(
        required=True,
    )
    model = forms.ChoiceField(
        required=True,
    )    
    batch_size = forms.IntegerField(
        min_value=1,
        required=True,
        initial=500
    )
    epochs = forms.IntegerField(
        min_value=1,
        required=True,
        initial=10,
    )
    verbose = forms.ChoiceField(
        choices = [(0, '0'), (1, '1'), (2, '2')],
        required=True,
    )
    validation_split = forms.DecimalField(
        min_value=0,
        max_value=1,
        initial=0.05,
        decimal_places=2,
    )


