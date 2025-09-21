# %pip install google-cloud-aiplatform kfp

import kfp
from kfp.v2 import dsl
from kfp.v2.dsl import (
    component,
    Input,
    Output,
    Model,
    Metrics,
    Artifact
)
from google.cloud import aiplatform
from typing import NamedTuple  # <-- 1. IMPORT NAMEDTUPLE

# --- 1. Define Your Project Constants ---

PROJECT_ID = "gke-dja-demo"
REGION = "us-central1"
PIPELINE_BUCKET_URI = "gs://gke-dja-sample-images/artifacts"
YOUR_GCS_DATA_PATH = "gs://gke-dja-sample-images/workshop/flowers-data-jpgs"

# --- 2. Define the Fine-Tuning Component ---
# (This component is UNCHANGED from our last step)
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
    model_output: Output[Model],
):
    import tensorflow as tf
    import os
    IMG_SIZE = (224, 224)
    AUTOTUNE = tf.data.AUTOTUNE

    # Load Data
    train_dir = os.path.join(dataset_gcs_path, "train")
    validation_dir = os.path.join(dataset_gcs_path, "validation")
    print(f"Loading training data from: {train_dir}")
    base_train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=IMG_SIZE, batch_size=batch_size, label_mode="categorical"
    )
    class_names = base_train_ds.class_names
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")
    train_dataset = base_train_ds.prefetch(buffer_size=AUTOTUNE)
    print(f"Loading validation data from: {validation_dir}")
    base_val_ds = tf.keras.utils.image_dataset_from_directory(
        validation_dir, image_size=IMG_SIZE, batch_size=batch_size, label_mode="categorical", shuffle=False
    )
    validation_dataset = base_val_ds.prefetch(buffer_size=AUTOTUNE)

    # Build Model
    print("Building EfficientNetV2-S model...")
    base_model = tf.keras.applications.EfficientNetV2S(
        input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet"
    )
    base_model.trainable = False
    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = tf.keras.applications.efficientnet_v2.preprocess_input(inputs) 
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    # Stage 1: Train Head
    print("--- Starting Stage 1: Feature Extraction (Training Head) ---")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy", metrics=["accuracy"]
    )
    model.fit(
        train_dataset, epochs=epochs, validation_data=validation_dataset
    )

    # Stage 2: Fine-Tuning
    if fine_tune_epochs > 0:
        print("--- Starting Stage 2: Fine-Tuning (Unfreezing Base Model) ---")
        base_model.trainable = True
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_learning_rate),
            loss="categorical_crossentropy", metrics=["accuracy"]
        )
        model.fit(
            train_dataset, epochs=epochs + fine_tune_epochs, initial_epoch=epochs, validation_data=validation_dataset
        )
    
    # Save Model
    print(f"Exporting SavedModel to artifact path: {model_output.path}")
    model.export(model_output.path) 
    print("Model export complete.")


# --- 3. Define the Evaluation Component (CORRECTED SIGNATURE) ---
@component(
    base_image="tensorflow/tensorflow:latest",
    packages_to_install=["google-cloud-storage"],
)
def evaluate_model(
    dataset_gcs_path: str,
    batch_size: int,
    model_input: Input[Model],
    # We also pass 'model_metrics' to log to the UI, but we return the dict
    model_metrics: Output[Metrics], 
) -> NamedTuple("EvaluateOutputs", [("metrics_dict", dict)]): 
    """
    Loads a SavedModel, evaluates it, logs metrics to the UI,
    and returns the metrics as a dictionary.
    """
    import tensorflow as tf
    import os
    
    print("--- Starting Evaluation Component (Keras 3) ---")
    IMG_SIZE = (224, 224)
    AUTOTUNE = tf.data.AUTOTUNE

    # --- 1. Load Test Data ---
    test_dir = os.path.join(dataset_gcs_path, "test")
    print(f"Loading test data from: {test_dir}")
    test_dataset = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False
    ).prefetch(buffer_size=AUTOTUNE)

    # --- 2. Load and Wrap Model ---
    print(f"Loading SavedModel as TFSMLayer from: {model_input.path}")
    sm_layer = tf.keras.layers.TFSMLayer(
        model_input.path, 
        call_endpoint='serving_default'
    )
    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), dtype=tf.float32)
    output_dict = sm_layer(inputs)
    outputs = list(output_dict.values())[0]
    eval_model = tf.keras.Model(inputs, outputs)
    
    print("Compiling wrapper model for evaluation...")
    eval_model.compile(loss="categorical_crossentropy", metrics=["accuracy"])
    
    # --- 3. Evaluate ---
    print("Evaluating model...")
    results = eval_model.evaluate(test_dataset, return_dict=True)
    print(f"Evaluation results: {results}")
    
    # --- 4. Log Metrics to UI (Good practice) ---
    model_metrics.log_metric("accuracy", results["accuracy"])
    model_metrics.log_metric("loss", results["loss"])
    print("Metrics logged to UI.")

    # --- 5. Return the results dict (THE FIX) ---
    # This dictionary will be passed as a Parameter to the next step.
    return (results,)


# --- 4. Define the Deployment Component (CORRECTED SIGNATURE) ---
@component(
    packages_to_install=["google-cloud-aiplatform"],
)
def deploy_model_to_vertex(
    project: str,
    region: str,
    model_display_name: str,
    endpoint_name: str,
    model_input: Input[Model],
    metrics_dict: dict,  # <-- FIX 2: We now accept a Python 'dict'
    min_accuracy_threshold: float,
) -> str:
    from google.cloud import aiplatform
    import json # No longer needed, but good to keep

    aiplatform.init(project=project, location=region)
    
    # --- 1. Check evaluation metrics (NOW FROM THE DICT) ---
    print("Checking evaluation metrics from input dictionary...")
    
    # --- THIS IS THE FIX ---
    # No file to open! KFP passes the Python dict directly.
    accuracy = metrics_dict['accuracy']
    # --- END OF FIX ---
    
    print(f"Model accuracy: {accuracy}")
    
    if accuracy < min_accuracy_threshold:
        print(f"Model accuracy {accuracy} is below threshold {min_accuracy_threshold}. Skipping deployment.")
        return ""

    print("Model accuracy is above threshold. Deploying...")

    # --- 2. Upload Model ---
    print(f"Uploading model '{model_display_name}' to Model Registry...")
    serving_image = "us-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-11:latest"
    model = aiplatform.Model.upload(
        display_name=model_display_name,
        artifact_uri=model_input.path,
        serving_container_image_uri=serving_image,
        sync=True
    )
    print(f"Model uploaded: {model.resource_name}")
    
    # --- 3. Create or Get Endpoint ---
    print(f"Creating or getting endpoint: {endpoint_name}")
    try:
        endpoint = aiplatform.Endpoint.create(
            display_name=endpoint_name,
            project=project,
            location=region
        )
        print(f"Created new endpoint: {endpoint.resource_name}")
    except Exception as e:
        if "already exists" in str(e):
            print("Endpoint already exists. Fetching.")
            endpoints = aiplatform.Endpoint.list(
                filter=f'display_name="{endpoint_name}"',
                project=project,
                location=region
            )
            endpoint = endpoints[0]
            print(f"Found existing endpoint: {endpoint.resource_name}")
        else:
            raise e

    # --- 4. Deploy Model to Endpoint ---
    print(f"Deploying model to endpoint {endpoint.resource_name}...")
    model.deploy(
        endpoint=endpoint,
        deployed_model_display_name=model_display_name,
        machine_type="g2-standard-4",
        accelerator_type="NVIDIA_L4",
        accelerator_count=1,
        traffic_split={"0": 100},
        sync=True
    )

    print(f"Model deployed to endpoint: {endpoint.resource_name}")
    return endpoint.resource_name

# --- 5. Define the Full End-to-End Pipeline (CORRECTED) ---
@dsl.pipeline(
    name="step-3-finetune-evaluate-deploy-pipeline",
    description="Full E2E pipeline: Trains, evaluates, and deploys the model.",
    pipeline_root=PIPELINE_BUCKET_URI,
)
def finetune_evaluate_deploy_pipeline(
    gcs_data_path: str,
    model_name: str = "efficientnet-flower-classifier",
    endpoint_name: str = "efficientnet-flower-endpoint",
    min_accuracy: float = 0.80,
    epochs: int = 1,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    fine_tune_epochs: int = 1,
    fine_tune_learning_rate: float = 1e-5
):
    
    # --- Step 1: Train ---
    train_op = train_efficientnet_v2s(
        dataset_gcs_path=gcs_data_path,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        fine_tune_epochs=fine_tune_epochs,
        fine_tune_learning_rate=fine_tune_learning_rate,
    )
    train_op.set_cpu_limit('8')\
            .set_memory_limit('16G')\
            .add_node_selector_constraint('NVIDIA_L4').set_accelerator_limit(1)

    # --- Step 2: Evaluate ---
    evaluate_op = evaluate_model(
        dataset_gcs_path=gcs_data_path,
        batch_size=batch_size,
        model_input=train_op.outputs["model_output"],
    )
    
    # --- Step 3: Deploy (CORRECTED) ---
    deploy_op = deploy_model_to_vertex(
        project=PROJECT_ID,
        region=REGION,
        model_display_name=model_name,
        endpoint_name=endpoint_name,
        model_input=train_op.outputs["model_output"],
        # --- THIS IS THE FIX ---
        # We pass the returned 'dict' (which KFP calls 'output')
        # to the 'metrics_dict' parameter.
        metrics_dict=evaluate_op.outputs["metrics_dict"],
        # --- END OF FIX ---
        min_accuracy_threshold=min_accuracy,
    )
    deploy_op.after(evaluate_op)

# --- 6. Compile and Run the Pipeline ---
# (This block is UNCHANGED)
from kfp.v2 import compiler

PIPELINE_JSON = "finetune_evaluate_deploy_pipeline.json"
compiler.Compiler().compile(
    pipeline_func=finetune_evaluate_deploy_pipeline,
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
        display_name="step-3-e2e-deploy-run-v3", # New display name
        template_path=PIPELINE_JSON,
        pipeline_root=PIPELINE_BUCKET_URI,
        enable_caching=True, # Caching is enabled
        parameter_values={
            "gcs_data_path": YOUR_GCS_DATA_PATH,
            "epochs": 1,
            "fine_tune_epochs": 1,
            "min_accuracy": 0.50 
        }
    )
    print("Submitting full end-to-end pipeline job...")
    job.run()
    print("Pipeline job submitted. Check the Vertex AI Pipelines UI.")