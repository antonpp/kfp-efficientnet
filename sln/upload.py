# %pip install google-cloud-storage tensorflow tensorflow-datasets

import tensorflow as tf
import tensorflow_datasets as tfds
from google.cloud import storage
import logging
import os

# --- 1. SET YOUR CONFIGURATION HERE ---

# The GCS bucket name (without gs://) where you want to upload the JPGs
DATA_BUCKET_NAME = "gke-dja-sample-images" 

# The folder path (prefix) within the bucket
GCS_PREFIX = "workshop/flowers-data-jpgs"

# -------------------------------------

# Set up logging to see progress
logging.basicConfig(level=logging.INFO, force=True, format='%(asctime)s - %(levelname)s - %(message)s')

def upload_data_to_gcs(bucket_name, gcs_prefix):
    """
    Downloads 'tf_flowers', converts to JPEG, and uploads to GCS.
    """
    
    # --- 1. Init GCS Client ---
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        logging.info(f"Successfully connected to GCS bucket: {bucket_name}")
        logging.info(f"Uploading files to: gs://{bucket_name}/{gcs_prefix}")
    except Exception as e:
        logging.error(f"Failed to connect to GCS bucket: {e}")
        logging.error("Please ensure you are authenticated (e.g., 'gcloud auth application-default login') and the bucket exists.")
        return

    # --- 2. Load dataset info to get class names ---
    logging.info("Loading 'tf_flowers' dataset info to get class names...")
    try:
        _, info = tfds.load('tf_flowers', with_info=True, split='train[:1%]')
        class_names = info.features['label'].names
        logging.info(f"Found {len(class_names)} class names: {class_names}")
    except Exception as e:
        logging.error(f"Failed to load TFDS dataset info: {e}")
        return

    # --- 3. Define splits and process each one ---
    splits_to_process = {
        'train': 'train[:80%]',      # 2936 images
        'validation': 'train[80%:90%]', # 367 images
        'test': 'train[90%:]'        # 367 images
    }

    total_uploaded = 0
    for split_name, split_range in splits_to_process.items():
        logging.info(f"--- Processing split: {split_name} ({split_range}) ---")
        
        try:
            # Load the full split
            ds = tfds.load('tf_flowers', split=split_range, as_supervised=True)
            
            count = 0
            for i, (image, label) in enumerate(ds):
                # Get the string name for the class (e.g., 'daisy')
                class_name = class_names[label.numpy()]
                
                # Convert the image tensor to JPEG bytes
                # Use quality=95 for high-quality JPGs
                jpeg_bytes = tf.io.encode_jpeg(image, quality=95)
                
                # Define the GCS blob path
                # e.g., 'workshop/flowers-data-jpgs/train/daisy/img_1.jpg'
                blob_path = os.path.join(gcs_prefix, split_name, class_name, f"img_{i}.jpg")
                
                # Create blob and upload from memory
                blob = bucket.blob(blob_path)
                blob.upload_from_string(
                    jpeg_bytes.numpy(),
                    content_type='image/jpeg'
                )
                
                count += 1
                if (i + 1) % 200 == 0:
                    logging.info(f"Uploaded {i+1} images for {split_name}...")
            
            logging.info(f"Completed split {split_name}. Uploaded {count} images.")
            total_uploaded += count
            
        except Exception as e:
            logging.error(f"Error processing split {split_name}: {e}")
            break # Stop if one split fails

    logging.info(f"--- UPLOAD COMPLETE ---")
    logging.info(f"Total images uploaded: {total_uploaded}")
    logging.info(f"Data is available at: gs://{bucket_name}/{gcs_prefix}")

# --- 4. Execute the function ---
if __name__ == "__main__":
    # Check if running in a notebook (which doesn't have __name__ == "__main__")
    # or as a script. This makes it safe to run in either.
    try:
        get_ipython()
        # We are in a notebook, just run the function
        upload_data_to_gcs(DATA_BUCKET_NAME, GCS_PREFIX)
    except NameError:
        # We are running as a script
        if DATA_BUCKET_NAME == "your-data-bucket":
            print("Please update DATA_BUCKET_NAME and GCS_PREFIX at the top of the script.")
        else:
            upload_data_to_gcs(DATA_BUCKET_NAME, GCS_PREFIX)