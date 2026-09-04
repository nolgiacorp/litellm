from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues

from ..base_llm.base_utils import BaseLLMModelInfo
from ..base_llm.chat.transformation import BaseLLMException

TOPAZ_IMAGE_VARIATION_MODELS = (
    "Standard V2",
    "Low Resolution V2",
    "CGI",
    "High Resolution V2",
    "Text Refine",
)

# Topaz video enhancement codes accepted by POST /video/express. Kept here rather than in
# the videos transformation so model discovery can list them without importing that module.
TOPAZ_VIDEO_MODELS = frozenset(
    (
        "aaa-9",
        "aaa-10",
        "ahq-12",
        "aion-1",
        "alq-13",
        "alqs-2",
        "amq-13",
        "amqs-2",
        "color-1",
        "ddv-3",
        "dtd-4",
        "dtds-2",
        "dtv-4",
        "dtvs-2",
        "ganim-1",
        "gcg-5",
        "ghq-5",
        "hyp-1",
        "hyp-2",
        "iris-2",
        "iris-3",
        "nxf-1",
        "nxl-1",
        "nyx-3",
        "pnat-1",
        "prob-4",
        "rhea-1",
        "sl-1",
        "slf-1",
        "slf-2",
        "slhq-1",
        "slm-1",
        "slp-2",
        "slp-2.5",
        "thd-3",
        "thf-4",
        "thm-2",
        "wonder-1",
    )
)


class TopazException(BaseLLMException):
    pass


class TopazModelInfo(BaseLLMModelInfo):
    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:
        if api_key is None:
            raise ValueError("API key is required for Topaz image variations. Set via `TOPAZ_API_KEY` or `api_key=..`")
        return {
            # "Content-Type": "multipart/form-data",
            "Accept": "image/jpeg",
            "X-API-Key": api_key,
        }

    def get_models(self, api_key: str | None = None, api_base: str | None = None) -> list[str]:
        return [  # mutable-ok: BaseLLMModelInfo contract returns List[str]
            f"topaz/{model}" for model in (*TOPAZ_IMAGE_VARIATION_MODELS, *sorted(TOPAZ_VIDEO_MODELS))
        ]

    @staticmethod
    def get_api_key(api_key: str | None = None) -> str | None:
        return api_key or get_secret_str("TOPAZ_API_KEY")

    @staticmethod
    def get_api_base(api_base: str | None = None) -> str | None:
        return api_base or get_secret_str("TOPAZ_API_BASE") or "https://api.topazlabs.com"

    @staticmethod
    def get_base_model(model: str) -> str:
        return model
