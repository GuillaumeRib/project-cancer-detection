import os
# $WIPE_BEGIN
from prefect.executors import LocalDaskExecutor
from prefect.run_configs import LocalRun
# $WIPE_END

from project_cancer_detection.flow.ml_flow import build_flow
flow = build_flow()

mlflow_experiment = os.environ.get("MLFLOW_EXPERIMENT")
prefect_backend = os.environ.get("PREFECT_BACKEND")

# $CODE_BEGIN
# OPTIONAL: Configure parallel task executor
flow.executor = LocalDaskExecutor()
# $CODE_END

# In dev mode, `make run_workflow` will run all tasks directly on your terminal
if prefect_backend == "development":
    flow.visualize()
    flow.run(parameters=dict(experiment=mlflow_experiment))

# In prod mode, `make run_workflow` only send a "snapshot" of your python code to Prefect (but does not executes it)
elif prefect_backend == "production":

    # dotenv is needed here to force sending the env values of your `.env` file to Prefect at each registry. Otherwise, Prefect caches the env variables and never updates them.
    from dotenv import dotenv_values
    env_dict = dotenv_values(".env")
    flow.run_config = LocalRun(env=env_dict)

    flow.register("project-cancer-detection")

else:
    raise ValueError(f"{prefect_backend} is not a valid value for PREFECT_BACKEND")
