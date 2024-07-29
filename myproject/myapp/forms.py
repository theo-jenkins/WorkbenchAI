from django import forms
from django.core.exceptions import ValidationError
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm, UserChangeForm
from .models import CustomUser
from .site_functions import get_uploaded_files

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
FEATURE_ENG_CHOICES = [
    ('handle_missing', 'Handle missing values'),
    ('normalize', 'Normalize'),
    ('standardize', 'Standardize'),
]
DATASET_TYPES = [
        ('features', 'Features (Input Data)'),
        ('outputs', 'Targets (Output Data)')
    ]

class ProcessDataForm(forms.Form):
    db_title = forms.CharField(
        required=True,
    )
    comment = forms.CharField(
        widget=forms.Textarea,
        required=False,
    )
    dataset_type = forms.ChoiceField(
        required=True,
        choices=DATASET_TYPES,
    )
    files = forms.MultipleChoiceField(
        widget=forms.CheckboxSelectMultiple,
        required=True,
    )
    features = forms.IntegerField(
        required=True,
        initial=1,
        min_value=1,
    )
    start_row = forms.IntegerField(
        initial=0,
    )
    end_row = forms.IntegerField(
        required=True,
    )

    def __init__(self, *args, **kwargs):
        feature_count = kwargs.pop('feature_count', 1)
        common_columns = kwargs.pop('common_columns', [])
        super(ProcessDataForm, self).__init__(*args, **kwargs)

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


LAYER_TYPE_CHOICES = [
    ('dense', 'Dense'),
]
ACTIVATION_CHOICES = [
    ('relu', 'ReLU'),
    ('sigmoid', 'Sigmoid'),
    ('softmax', 'Softmax'),
    ('softplus', 'Softplus'),
    ('softsign', 'Softsign'),
    ('tanh', 'Tanh'),
    ('selu', 'SELU'),
    ('elu', 'ELU'),
    ('exponential', 'Exponential'),
    ('hard_sigmoid', 'Hard Sigmoid'),
    ('linear', 'Linear'),
    ('swish', 'Swish'),
    ('gelu', 'GELU')
]
OPTIMIZER_CHOICES = [
        ('adam', 'Adam'),
        ('sgd', 'SGD'),
        ('rmsprop', 'RMSprop'),
        ('adagrad', 'Adagrad'),
        ('adadelta', 'Adadelta'),
        ('adamax', 'Adamax'),
        ('nadam', 'Nadam')
]
LOSS_CHOICES = [
        ('categorical_crossentropy', 'Categorical Cross-Entropy'),
        ('binary_crossentropy', 'Binary Cross-Entropy'),
        ('mean_squared_error', 'Mean Squared Error'),
        ('mean_absolute_error', 'Mean Absolute Error'),
        ('hinge', 'Hinge Loss'),
        ('sparse_categorical_crossentropy', 'Sparse Categorical Cross-Entropy'),
]
METRIC_CHOICES = [
        ('accuracy', 'Accuracy'),
        ('precision', 'Precision'),
        ('recall', 'Recall'),
        ('f1_score', 'F1 Score'),
        ('mean_squared_error', 'Mean Squared Error'),
        ('mean_absolute_error', 'Mean Absolute Error')
    ]
class BuildModelForm(forms.Form):
    model_title = forms.CharField(
        required=True,
        label='Model Title',
    )
    comment = forms.CharField(
        widget=forms.Textarea(),
        required=False,
        label='Comment',
    )
    features = forms.IntegerField(
        required=True,
        initial=1,
        min_value=1,
        label='Features',
    )
    feature_layer_type = forms.ChoiceField(
        choices=LAYER_TYPE_CHOICES,
        required=True,
        label='Layer Type'
    )
    feature_activation = forms.ChoiceField(
        choices=ACTIVATION_CHOICES,
        required=True,
        label='Activation Function'
    )
    outputs = forms.IntegerField(
        required=True,
        initial=1,
        min_value=1,
        label='Outputs',
    )
    output_layer_type = forms.ChoiceField(
        choices=LAYER_TYPE_CHOICES,
        required=True,
        label='Layer Type'
    )
    output_activation = forms.ChoiceField(
        choices=ACTIVATION_CHOICES,
        required=True,
        label='Activation Function'
    )
    hidden_layers = forms.IntegerField(
        required=True,
        initial=1,
        min_value=0,
        label='Hidden Layers',
    )

    def __init__(self, *args, **kwargs):
        hidden_layer_count = kwargs.pop('hidden_layer_count', 1)
        super(BuildModelForm, self).__init__(*args, **kwargs)
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
                choices=ACTIVATION_CHOICES,
                required=True,
            )


    optimizer = forms.ChoiceField(
        choices=OPTIMIZER_CHOICES,
        required=True,
        label='Optimizer'
    )
    loss = forms.ChoiceField(
        choices=LOSS_CHOICES,
        required=True,
        label='Loss Function'
    )
    metrics = forms.MultipleChoiceField(
        widget=forms.CheckboxSelectMultiple,
        choices=METRIC_CHOICES,
        required=True,
        label='Metrics'
    )

class TrainModelForm(forms.Form):
    feature_dataset = forms.ChoiceField(
        required=True
    )
    training_dataset = forms.ChoiceField(
        required=True,
    )
    model = forms.ChoiceField(
        required=True,
    )
    comment = forms.CharField(
        widget=forms.Textarea(),
        required=False,
    )
    batch_size = forms.IntegerField(
        min_value=1,
        required=True,
        initial=500
    )
    epochs = forms.IntegerField(
        min_value=1,
        required=True,
        initial=60,
    )
    verbose = forms.ChoiceField(
        required=True,
    )
    validation_split = forms.DecimalField(
        min_value=0,
        max_value=1,
        initial=0.05,
    )

class ProcessTickerForm(forms.Form):
    ticker = forms.CharField(
        required=True,        
    )


