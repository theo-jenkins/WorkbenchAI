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
    max_size_mb = 100 # Maximum file size in MB
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
    dataset_type = forms.ChoiceField(
        required=True
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
        help_text='Enter the title of the model.'
    )
    comment = forms.CharField(
        widget=forms.Textarea(),
        required=False,
        label='Comment',
        help_text='Optional: Add any comments about the model.'
    )
    features = forms.IntegerField(
        required=True,
        initial=1,
        min_value=1,
        label='Features',
        help_text='Enter the number of features for the input layer.'
    )
    outputs = forms.IntegerField(
        required=True,
        initial=1,
        min_value=1,
        label='Outputs',
        help_text='Enter the number of outputs for the output layer.'
    )
    hidden_layers = forms.IntegerField(
        required=True,
        initial=1,
        min_value=1,
        label='Hidden Layers',
        help_text='Enter the number of hidden layers.'
    )
    layer_type = forms.ChoiceField(
        choices=LAYER_TYPE_CHOICES,
        required=True,
        label='Layer Type'
    )
    activation = forms.ChoiceField(
        choices=ACTIVATION_CHOICES,
        required=True,
        label='Activation Function'
    )
    nodes = forms.IntegerField(
        required=True,
        min_value=1,
        initial=32,
        label='Nodes',
        help_text='Enter the number of nodes for each layer.'
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
    validation_split = forms.IntegerField(
        min_value=0,
        max_value=1,
        initial=0.05,
    )

class ProcessTickerForm(forms.Form):
    ticker = forms.CharField(
        required=True,        
    )