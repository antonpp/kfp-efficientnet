# %pip install google-cloud-aiplatform kfp

import kfp
from kfp.v2 import dsl
from kfp.v2.dsl import (
    component,
    Input,
    Output,
    Model,
    Metrics, # Not used in this pipeline, but good to import
    Artifact
)
from google.cloud import aiplatform

# --- 1. Define Your Project Constants ---

# Your Google Cloud Project ID
PROJECT_ID = "gke-dja-demo"

# The region for your Vertex AI resources
REGION = "us-central1"

# The GCS bucket for storing pipeline artifacts (logs, trained model, etc.)
PIPELINE_BUCKET_URI = "gs://gke-dja-sample-images/artifacts"


# --- 2. Define the Fine-Tuning Component (CORRECTED) ---
@component(
    base_image="tensorflow/tensorflow:latest",
    packages_to_install=["google-cloud-storage"],
)
def train_efficientnet_v2s(
    dataset_gcs_path: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    fine_tune_epochs: int,
    fine_tune_learning_rate: float,
    model_output: ???, # TODO
):
    """
    Fine-tunes an EfficientNetV2-S model using JPGs from a GCS path.
    """
    import tensorflow as tf
    import os
    
    IMG_SIZE = (224, 224)
    AUTOTUNE = tf.data.AUTOTUNE

    # --- 1. Load Data from GCS (CORRECTED BLOCK) ---
    train_dir = os.path.join(dataset_gcs_path, "train")
    validation_dir = os.path.join(dataset_gcs_path, "validation")

    print(f"Loading training data from: {train_dir}")
    # Create the base dataset FIRST, without prefetch
    base_train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode="categorical"
    )
    
    # NOW, get the class names from the base dataset
    class_names = base_train_ds.class_names
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")

    # THEN, apply prefetch to the base dataset
    train_dataset = base_train_ds.prefetch(buffer_size=AUTOTUNE)

    print(f"Loading validation data from: {validation_dir}")
    # We don't access class_names here, but it's good practice to be consistent
    base_val_ds = tf.keras.utils.image_dataset_from_directory(
        validation_dir,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False
    )
    validation_dataset = base_val_ds.prefetch(buffer_size=AUTOTUNE)
    # --- END OF CORRECTED BLOCK ---

    # --- 2. Build the Model ---
    print("Building EfficientNetV2-S model...")
    base_model = tf.keras.applications.EfficientNetV2S(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False
    
    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = tf.keras.applications.efficientnet_v2.preprocess_input(inputs) 
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    # Use the 'num_classes' variable we correctly retrieved
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    # --- 3. Stage 1: Train the Classifier Head ---
    print("--- Starting Stage 1: Feature Extraction (Training Head) ---")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=validation_dataset
    )

    # --- 4. Stage 2: Fine-Tuning ---
    if fine_tune_epochs > 0:
        print("--- Starting Stage 2: Fine-Tuning (Unfreezing Base Model) ---")
        base_model.trainable = True
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        model.fit(
            train_dataset,
            epochs=epochs + fine_tune_epochs,
            initial_epoch=epochs,
            validation_data=validation_dataset
        )
    
    # --- 5. Save the final model ---
    print(f"Saving model to artifact path: {model_output.path}")
    # Use model.export() to save in the SavedModel directory format
    # This is what Vertex AI Prediction needs
    model.export(model_output.path) 
    
    print("Model export complete.")


# --- 2. Define the Pipeline (with L4) ---
@dsl.pipeline(
    name="step-1-finetune-only-pipeline-l4",
    description="Runs only the fine-tuning component with an L4 GPU.",
    pipeline_root=PIPELINE_BUCKET_URI,
)
def finetune_only_pipeline(
    gcs_data_path: str,
    epochs: int = 1,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    fine_tune_epochs: int = 1,
    fine_tune_learning_rate: float = 1e-5
):
    
    # Call the training component
    train_op = train_efficientnet_v2s(
        dataset_gcs_path=gcs_data_path,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        fine_tune_epochs=fine_tune_epochs,
        fine_tune_learning_rate=fine_tune_learning_rate,
    )
    
    # --- THIS IS THE UPDATED PART ---
    # We've changed 'NVIDIA_TESLA_T4' to 'NVIDIA_L4'
    train_op.set_cpu_limit('8')\
            .set_memory_limit('16G')\
            .add_node_selector_constraint('NVIDIA_L4').set_accelerator_limit(1)
    # --- END OF UPDATED PART ---

# --- 3. Compile and Run the Pipeline ---

# Make sure your project constants are set
PROJECT_ID = "gke-dja-demo"
REGION = "us-central1"
PIPELINE_BUCKET_URI = "gs://gke-dja-sample-images/artifacts"
YOUR_GCS_DATA_PATH = "gs://gke-dja-sample-images/workshop/flowers-data-jpgs"

from kfp.v2 import compiler

PIPELINE_JSON = "finetune_only_pipeline_l4.json"
compiler.Compiler().compile(
    pipeline_func=finetune_only_pipeline,
    package_path=PIPELINE_JSON,
)

print(f"Pipeline compiled to {PIPELINE_JSON}")

aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=PIPELINE_BUCKET_URI)

if YOUR_GCS_DATA_PATH == "gs://your-data-bucket/workshop/flowers-data-jpgs":
    print("="*80)
    print("ERROR: Please update 'YOUR_GCS_DATA_PATH' with your actual GCS bucket path.")
    print("="*80)
else:
    job = aiplatform.PipelineJob(
        display_name="step-1-finetune-run-l4", # Updated display name
        template_path=PIPELINE_JSON,
        pipeline_root=PIPELINE_BUCKET_URI,
        enable_caching=True,
        parameter_values={ # TODO
            "???": ???,
            "???": ???,
            ...
        }
    )
    print("Submitting pipeline job with L4 GPU request...")
    job.run()
    print("Pipeline job submitted. Check the Vertex AI Pipelines UI.")