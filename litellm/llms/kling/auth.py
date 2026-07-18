import time

import jwt

import litellm
from litellm.secret_managers.main import get_secret_str

_KLING_JWT_TTL_SECONDS = 1800
_KLING_JWT_LEEWAY_SECONDS = 5


def resolve_kling_api_key(api_key: str | None) -> str:
    resolved = api_key or litellm.api_key or get_secret_str("KLING_API_KEY")
    if not resolved:
        raise ValueError(
            "Kling API key is required. Set KLING_API_KEY (format 'AccessKey:SecretKey') "
            "environment variable or pass the api_key parameter."
        )
    return resolved


def split_access_secret(api_key: str) -> tuple[str, str]:
    access_key, separator, secret_key = api_key.partition(":")
    if not separator or not access_key or not secret_key:
        raise ValueError("KLING_API_KEY must be in the form 'AccessKey:SecretKey' (two colon-separated parts).")
    return access_key, secret_key


def generate_kling_jwt(api_key: str) -> str:
    access_key, secret_key = split_access_secret(api_key)
    now = int(time.time())
    payload = {
        "iss": access_key,
        "exp": now + _KLING_JWT_TTL_SECONDS,
        "nbf": now - _KLING_JWT_LEEWAY_SECONDS,
    }
    return jwt.encode(
        payload,
        secret_key,
        algorithm="HS256",
        headers={"alg": "HS256", "typ": "JWT"},
    )


def kling_auth_headers(api_key: str | None) -> dict:
    token = generate_kling_jwt(resolve_kling_api_key(api_key))
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
