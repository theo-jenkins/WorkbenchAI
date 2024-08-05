FEATURE_ENG_CHOICES = [
    ('handle_missing', 'Handle missing values'),
    ('normalize', 'Normalize'),
    ('standardize', 'Standardize'),
]
DATASET_TYPE_CHOICES = [
    ('features', 'Features (Input Data)'),
    ('outputs', 'Targets (Output Data)')
]
MODEL_TYPE_CHOICES = [
    ('sequential', 'Sequential'),
    ('LSTM', 'LSTM'),
]
LAYER_TYPE_CHOICES = [
    ('dense', 'Dense'),
]
ACTIVATION_TYPE_CHOICES = [
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