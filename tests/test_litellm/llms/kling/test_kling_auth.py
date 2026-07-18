import time

import jwt
import pytest

from litellm.llms.kling.auth import (
    generate_kling_jwt,
    kling_auth_headers,
    resolve_kling_api_key,
    split_access_secret,
)

ACCESS_KEY = "A" * 32
SECRET_KEY = "S" * 32
API_KEY = f"{ACCESS_KEY}:{SECRET_KEY}"


def test_split_access_secret_parses_two_parts():
    assert split_access_secret(API_KEY) == (ACCESS_KEY, SECRET_KEY)


@pytest.mark.parametrize("bad_key", ["no-colon", ":only-secret", "only-access:", ""])
def test_split_access_secret_rejects_malformed_key(bad_key):
    with pytest.raises(ValueError, match="AccessKey:SecretKey"):
        split_access_secret(bad_key)


def test_generate_kling_jwt_header_and_payload():
    before = int(time.time())
    token = generate_kling_jwt(API_KEY)
    after = int(time.time())

    header = jwt.get_unverified_header(token)
    assert header["alg"] == "HS256"
    assert header["typ"] == "JWT"

    payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    assert payload["iss"] == ACCESS_KEY
    assert before + 1800 <= payload["exp"] <= after + 1800
    assert before - 5 <= payload["nbf"] <= after - 5


def test_generate_kling_jwt_is_signed_with_secret_key():
    token = generate_kling_jwt(API_KEY)
    # Verifying with the access key (wrong secret) must fail.
    with pytest.raises(jwt.InvalidSignatureError):
        jwt.decode(token, ACCESS_KEY, algorithms=["HS256"])


def test_kling_auth_headers_bearer_token():
    headers = kling_auth_headers(API_KEY)
    assert headers["Authorization"].startswith("Bearer ")
    assert headers["Content-Type"] == "application/json"
    token = headers["Authorization"].split(" ", 1)[1]
    payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    assert payload["iss"] == ACCESS_KEY


def test_resolve_kling_api_key_prefers_explicit(monkeypatch):
    monkeypatch.delenv("KLING_API_KEY", raising=False)
    assert resolve_kling_api_key(API_KEY) == API_KEY


def test_resolve_kling_api_key_reads_env(monkeypatch):
    monkeypatch.setenv("KLING_API_KEY", API_KEY)
    assert resolve_kling_api_key(None) == API_KEY


def test_resolve_kling_api_key_raises_when_missing(monkeypatch):
    import litellm

    monkeypatch.delenv("KLING_API_KEY", raising=False)
    monkeypatch.setattr(litellm, "api_key", None)
    with pytest.raises(ValueError, match="Kling API key is required"):
        resolve_kling_api_key(None)
