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

# --- 1. Define Your Project Constants ---

PROJECT_ID = "gke-dja-demo"
REGION = "us-central1"
PIPELINE_BUCKET_URI = "gs://gke-dja-sample-images/artifacts"
YOUR_GCS_DATA_PATH = "gs://gke-dja-sample-images/workshop/flowers-data-jpgs"

# --- 2. Define the Fine-Tuning Component ---
# (This is the same, corrected component from the last step)
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
    """
    Fine-tunes an EfficientNetV2-S model using JPGs from a GCS path.
    """
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


# --- 3. Define the Evaluation Component (CORRECTED) ---
@component(
    base_image="tensorflow/tensorflow:latest",
    packages_to_install=["google-cloud-storage"],
)
def evaluate_model(
    dataset_gcs_path: str,
    batch_size: int,
    model_input: Input[Model],
    model_metrics: Output[Metrics],
):
    """
    Loads a SavedModel (as a TFSMLayer) and evaluates it.
    """
    import tensorflow as tf
    import os
    
    print("--- Starting Evaluation Component (Keras 3) ---")
    IMG_SIZE = (224, 224)
    AUTOTUNE = tf.data.AUTOTUNE

    # --- 1. Load Test Data from GCS ---
    test_dir = os.path.join(dataset_gcs_path, "test")
    print(f"Loading test data from: {test_dir}")

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False
    ).prefetch(buffer_size=AUTOTUNE)

    # --- 2. Load and Evaluate Model (CORRECTED for TFSMLayer output) ---
    print(f"Loading SavedModel as TFSMLayer from: {model_input.path}")
    
    sm_layer = tf.keras.layers.TFSMLayer(
        model_input.path, 
        call_endpoint='serving_default'
    )
    
    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), dtype=tf.float32)
    
    # --- THIS IS THE FIX ---
    # The TFSMLayer returns a dictionary of outputs (e.g., {'output_0': <tensor>}).
    # We need to extract the actual tensor to match our 'y_true' from the dataset.
    # Since our model has only one output, we can take the first value.
    output_dict = sm_layer(inputs)
    outputs = list(output_dict.values())[0]
    # --- END OF FIX ---
    
    # Create the new wrapper model
    eval_model = tf.keras.Model(inputs, outputs)
    
    # Compile the wrapper model
    print("Compiling wrapper model for evaluation...")
    eval_model.compile(loss="categorical_crossentropy", metrics=["accuracy"])
    
    # --- 3. Evaluate ---
    print("Evaluating model...")
    results = eval_model.evaluate(test_dataset, return_dict=True)
    
    print(f"Evaluation results: {results}")
    
    # --- 4. Log Metrics to Vertex AI ---
    model_metrics.log_metric("accuracy", results["accuracy"])
    model_metrics.log_metric("loss", results["loss"])
    print("Metrics logged.")

# --- 4. Define the NEW Pipeline (Train + Evaluate) ---
@dsl.pipeline(
    name="step-2-finetune-and-evaluate-pipeline",
    description="Trains and then evaluates the model.",
    pipeline_root=PIPELINE_BUCKET_URI,
)
def finetune_and_evaluate_pipeline(
    gcs_data_path: str,
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
    
    # Request the L4 GPU for the training step
    # This MUST match the previous run to get a cache hit
    #train_op.set_machine_type('n1-standard-4') \
    #        .set_accelerator_type('NVIDIA_L4') \
    #        .set_accelerator_limit(1)

    train_op.set_cpu_limit('8')\
            .set_memory_limit('16G')\
            .add_node_selector_constraint('NVIDIA_L4').set_accelerator_limit(1)

    # --- Step 2: Evaluate ---
    # This step takes the output of the training step as its input
    evaluate_op = evaluate_model(
        dataset_gcs_path=gcs_data_path,
        batch_size=batch_size,
        model_input=train_op.outputs["model_output"], # Connect the steps
    )

# --- 5. Compile and Run the Pipeline ---

from kfp.v2 import compiler

PIPELINE_JSON = "finetune_and_evaluate_pipeline.json"
compiler.Compiler().compile(
    pipeline_func=finetune_and_evaluate_pipeline,
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
        display_name="step-2-finetune-evaluate-run",
        template_path=PIPELINE_JSON,
        pipeline_root=PIPELINE_BUCKET_URI,
        enable_caching=True, # <-- Caching is enabled
        parameter_values={
            "gcs_data_path": YOUR_GCS_DATA_PATH,
            # We use the *exact same* parameters as the "quick run"
            # to ensure we get a cache hit for the training step.
            "epochs": 1,
            "fine_tune_epochs": 1,
        }
    )
    print("Submitting pipeline job...")
    job.run()
    print("Pipeline job submitted. Check the Vertex AI Pipelines UI.")