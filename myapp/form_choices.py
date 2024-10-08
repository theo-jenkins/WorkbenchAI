ACCOUNT_TYPE_CHOICES = [
    ('developer', 'Developer'),
]
FEATURE_ENG_CHOICES = [
    ('date_col', 'Convert to datetime object'),
    ('handle_missing', 'Handle missing values'),
    ('normalize', 'Normalize'),
    ('standardize', 'Standardize'),
]
DATASET_FORM_CHOICES = [
    ('tabular', 'Tabular'),
    ('ts', 'Timeseries')
]
DATASET_TYPE_CHOICES = [
    ('features', 'Features (Input Data)'),
    ('outputs', 'Targets (Output Data)')
]
AGGREGATION_FREQUENCY_CHOICES = [
    ('H', 'Hourly'),
    ('D', 'Daily'),
    ('W', 'Weekly'),
    ('M', 'Monthly'),
    ('Y', 'Yearly'),
]
MODEL_FORM_CHOICES = [
    ('sequential', 'Sequential'),
    ('xgboost', 'XGBoost'),
    ('mamba', 'Mamba SSM'),
]
LAYER_TYPE_CHOICES = [
    ('dense', 'Dense (Fully Connected)'),
    ('LSTM', 'LSTM (Long Short-Term Memory)'),
    ('GRU', 'GRU (Gated Recurrent Unit)')
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