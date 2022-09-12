import mlflow

mlflow.set_tracking_uri("MLFLOW_TRACKING_URI")
mlflow.set_experiment(experiment_name="project-cancer-detection")
with mlflow.start_run():

    params = dict(batch_size=batch_size, epochs=epochs)
    metrics = dict(loss=model_outputs[0], accuracy=model_outputs[1])

    mlflow.log_params(params)
    mlflow.log_metrics(metrics)

    mlflow.keras.log_model(keras_model=model,
                        artifact_path="model",
                        keras_module="tensorflow.keras",
                        registered_model_name="cancer_detection_model")
