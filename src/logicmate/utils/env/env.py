from enum import Enum
import os
from dotenv import load_dotenv


def load_env_variables(env_path: str | None = None) -> None:
    """
    Load environment variables from a .env file.

    Args:
        env_path (str | None): Path to the .env file. If None, defaults to the .env file in the parent directory.
    Raises:
        FileNotFoundError: If the .env file is not found at the specified path.

    This function loads environment variables from a .env file using the `dotenv` library.
    """
    if env_path is None:
        base_dir: str = os.path.dirname(
            p=os.path.dirname(p=os.path.abspath(path=__file__))
        )
        env_path = os.path.join(base_dir, ".env")

    if not os.path.exists(path=env_path):
        raise FileNotFoundError(f".env file not found at {env_path}")

    load_dotenv(dotenv_path=env_path)


class EnvVariable(Enum):
    """
    Enum class for environment variables.

    Attributes:
        ROBOFLOW_API_KEY (str): The API key for Roboflow.
        ROBOFLOW_API_URL (str): The API URL for Roboflow.
        OPENAI_API_KEY (str): The API key for OpenAI.
    """

    ROBOFLOW_API_KEY: str = "ROBOFLOW_API_KEY"
    ROBOFLOW_API_URL: str = "ROBOFLOW_API_URL"
    OPENAI_API_KEY: str = "OPENAI_API_KEY"


def get_required_env_variables() -> dict[str, str]:
    """
    Get required environment variables.

    Returns:
        dict[str, str]: A dictionary containing the required environment variables.

    This function retrieves the required environment variables from the .env file.
    """
    env_vars: dict[str, str] = {
        EnvVariable.ROBOFLOW_API_KEY.value: os.getenv(
            key=EnvVariable.ROBOFLOW_API_KEY.value
        ),
        EnvVariable.ROBOFLOW_API_URL.value: os.getenv(
            key=EnvVariable.ROBOFLOW_API_URL.value
        ),
        EnvVariable.OPENAI_API_KEY.value: os.getenv(
            key=EnvVariable.OPENAI_API_KEY.value
        ),
    }
    return env_vars
