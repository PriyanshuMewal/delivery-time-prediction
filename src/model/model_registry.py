import json
from mlflow import MlflowClient
import logging
import mlflow
import dagshub
import mlflow.sklearn
import yaml

logger = logging.getLogger("model_registry")
logger.setLevel("DEBUG")

handler = logging.StreamHandler()
handler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

logger.addHandler(handler)


# Authenticate with dagshub:
def dagshub_authentication() -> None:
    logger.info("Authenticating with Dagshub ...")

    try:
        mlflow.set_tracking_uri("https://dagshub.com/PriyanshuMewal/delivery-time-prediction.mlflow")
        dagshub.init(repo_owner='PriyanshuMewal', repo_name='delivery-time-prediction', mlflow=True)

    except Exception as e:
        logger.error(f"Authentication Failed: {e}")
        raise

    logger.info("Dagshub Authentication Completed Successfully!")


def load_data(url: str) -> dict:
    logger.info(f"Loading model info from {url} ...")

    try:
        with open(url, mode="r") as file:
            model_info = json.load(file)

    except FileNotFoundError:
        logger.error(f"Model info not found: {url}")
        raise
    except Exception as e:
        logger.error(f"Failed to parse model info: {e}")
        raise


    if "version" not in model_info or "name" not in model_info:
        logger.error("Either model_version or model_name doesn't exist in model_info file.")
        raise KeyError("Few info is missing from model_info file.")

    if len(model_info) == 0:
        logger.error("Model info is empty check the evaluation code")
        raise ValueError("model_info is empty")

    logger.info("model_info loaded successfully.")
    return model_info


def load_params(param_url: str) -> dict:
    logger.info(f"Loading parameter from {param_url}.")

    try:
        with open(param_url, "r") as file:
            params = yaml.safe_load(file)
    except FileNotFoundError:
        logger.error(f"Params file not found: {param_url}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML file: {e}")
        raise

    logger.info("Parameters loaded successfully.")

    try:
        return params["model_registry"]
    except (TypeError, KeyError):
        logger.error("Missing 'model_registry' section in params.yaml")
        raise


def update_model(name: str, version: str) -> None:

    client = MlflowClient()

    # load parameter:
    params_url = "params.yaml"
    params = load_params(params_url)

    try:
        description = params["model_description"]
        tags = params["model_tags"]
        new_stage = params["model_stage"]
        alias = params["model_alias"]
    except KeyError as e:
        logger.error(f"Some info is missing: {e}")
        raise

    # add description to the model:
    client.update_model_version(
        name=name, version=version,
        description=description
    )
    logger.debug("Description added!")

    # add some tags to the model:
    for key, value in tags.items():

        client.set_model_version_tag(
            name=name, version=version,
            key=key, value=value,
        )
    logger.debug("Description added!")

    # Transition version stage:
    client.transition_model_version_stage(
        name=name, version=version,
        stage=new_stage,
    )
    logger.debug("Model version transition done.")

    # Declare it as challenger:
    client.set_registered_model_alias(
        name=name, version=version,
        alias=alias
    )


def main() -> None:

    # log everything
    logger.info("Updating the model's meta data from model registry ...")
    dagshub_authentication()

    # load model info:
    path = "reports/model_info.json"
    model_info = load_data(path)

    model_name = model_info["name"]
    version = model_info["version"]

    # Update model from model registry:
    update_model(model_name, version)

    logger.debug("Registered model meta data updated successfully!")


if __name__ == "__main__":
    main()