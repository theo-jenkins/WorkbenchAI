import os
import zipfile
import subprocess
import pandas as pd
from django.conf import settings
from django.core.exceptions import ValidationError
from .models import FileMetadata

# Function that returns the user uploaded files
def get_file_choices(user):
    user_files = FileMetadata.objects.filter(user=user)
    return user_files

# Function that checks if a file exists
def file_exists(uploaded_file):
    upload_dir = os.path.join(settings.USER_ROOT)
    file_path = os.path.join(upload_dir, uploaded_file.name)
    return os.path.exists(file_path)

# Function that retrieves the latest git commit information
def get_latest_commit_info():
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--date=format:'%a %b %d'", "--format=%s:%cd"],
            capture_output=True,
            text=True,
            check=True
        )
        commit_info = result.stdout.strip()
        message = (f'Latest commit: {commit_info}')
        return message
    except subprocess.CalledProcessError as e:
        print(f'Error while fetching commit info: {e}')
        return None

# Function that validates and saves the file to upload_dir and deletes the .zip if applicable
def upload_file(file):
    valid_extensions = ['.zip', '.csv']    
    # Check if the file has a valid extension
    if not any(file.name.endswith(ext) for ext in valid_extensions):
        raise ValidationError('Invalid file extension')
    
    try:
        # Ensure the uploaded_files directory exists
        upload_dir = os.path.join(settings.USER_ROOT)
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save the uploaded file to the directory
        file_path = os.path.join(upload_dir, file.name)
        with open(file_path, 'wb+') as destination:
            for chunk in file.chunks():
                destination.write(chunk)
                
        # Handles .zip files
        if file.name.endswith('.zip'):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                # Extract only .csv files
                for zip_info in zip_ref.infolist():
                    if zip_info.filename.endswith('.csv'):
                        zip_info.filename = os.path.basename(zip_info.filename)  # Removes any directory structure
                        zip_ref.extract(zip_info, upload_dir)
            os.remove(file_path)  # Remove the .zip file after extraction
            
        return file  # Return the file object for further use if needed
    except Exception as e:
        raise ValidationError("There was an error uploading the file. Please try again.")
    

# Function that finds the common columns between .csv files selected
# Returns list of common columns
def get_common_columns(file_ids):
    file_paths = FileMetadata.objects.filter(id__in=file_ids).values_list('file_path', flat=True)
    upload_dir = settings.USER_ROOT
    common_columns = None
    for file_path in file_paths:
        full_path = os.path.join(upload_dir, file_path)
        # Read only the first row to get the columns
        df = pd.read_csv(full_path, nrows=0)  # nrows=0 loads no data rows, only headers

        if common_columns is None:
            common_columns = set(df.columns)
        else:
            common_columns.intersection_update(df.columns)

        if not common_columns:
            break  # No need to proceed if there are no common columns left

    return sorted(common_columns) if common_columns else []

# Function finds the maximum rows in the .csv files selected
# Returns a integer for end_rows
def get_max_rows(file_ids):
    # Fetch the file paths associated with the selected file IDs
    file_paths = FileMetadata.objects.filter(id__in=file_ids).values_list('file_path', flat=True)

    # Compute the total number of rows across all files
    max_rows = sum(pd.read_csv(file_path).shape[0] for file_path in file_paths)

    return max_rows