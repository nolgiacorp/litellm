from openai.types.fine_tuning.fine_tuning_job import Hyperparameters
from pydantic import ConfigDict


class OpenAIFineTuningHyperparameters(Hyperparameters):
    model_config = ConfigDict(extra="allow")
