"""Thin boto3 wrapper for AWS Bedrock Runtime.

Credentials are sourced from the standard boto3 chain
(environment variables, ~/.aws/credentials, IAM role, etc.).
No credentials are hardcoded here.

AWS_REGION environment variable controls the region; defaults to us-east-1.
"""

import json
import logging
import os
import time

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

_DEFAULT_REGION = "us-east-1"
_DEFAULT_MODEL = "us.meta.llama3-1-8b-instruct-v1:0"


class BedrockClient:
    """Wrapper around the AWS Bedrock Runtime ``invoke_model`` API.

    Parameters
    ----------
    region:
        AWS region name. Reads from the ``AWS_REGION`` environment variable,
        falling back to *us-east-1*.
    model_id:
        Default Bedrock model ID to use when none is specified per call.
    """

    def __init__(
        self,
        region: str | None = None,
        model_id: str = _DEFAULT_MODEL,
    ) -> None:
        self.region = region or os.environ.get("AWS_REGION", _DEFAULT_REGION)
        self.default_model_id = model_id
        self._client = boto3.client("bedrock-runtime", region_name=self.region)
        logger.debug("BedrockClient initialised (region=%s, model=%s)", self.region, self.default_model_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_body(
        self,
        prompt: str,
        system: str,
        model_id: str,
        max_tokens: int,
    ) -> str:
        """Build the JSON body for a Llama3-family instruct model."""
        # Llama 3 instruct format used by Bedrock
        full_prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{prompt}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        body = {
            "prompt": full_prompt,
            "max_gen_len": max_tokens,
            "temperature": 0.0,
            "top_p": 0.9,
        }
        return json.dumps(body)

    def _parse_response(self, response: dict) -> str:
        """Extract the generated text from a Bedrock response object."""
        body_bytes = response["body"].read()
        data = json.loads(body_bytes)
        # Llama models return {"generation": "..."}
        return data.get("generation", "").strip()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def invoke(
        self,
        prompt: str,
        system: str,
        model_id: str | None = None,
        max_tokens: int = 256,
    ) -> str:
        """Invoke a Bedrock model once (no retry).

        Parameters
        ----------
        prompt:
            User-turn text.
        system:
            System-prompt text.
        model_id:
            Model ID to use; defaults to ``self.default_model_id``.
        max_tokens:
            Maximum tokens to generate.

        Returns
        -------
        str
            Raw generated text from the model.

        Raises
        ------
        ClientError
            Propagated from boto3 on non-throttling errors.
        """
        effective_model = model_id or self.default_model_id
        body = self._build_body(prompt, system, effective_model, max_tokens)
        response = self._client.invoke_model(
            modelId=effective_model,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        return self._parse_response(response)

    def invoke_with_retry(
        self,
        prompt: str,
        system: str,
        model_id: str | None = None,
        max_tokens: int = 256,
        max_retries: int = 3,
        base_delay: float = 2.0,
    ) -> str:
        """Invoke a Bedrock model with exponential backoff on throttling.

        Parameters
        ----------
        prompt:
            User-turn text.
        system:
            System-prompt text.
        model_id:
            Model ID to use; defaults to ``self.default_model_id``.
        max_tokens:
            Maximum tokens to generate.
        max_retries:
            Maximum number of retry attempts on ``ThrottlingException``.
        base_delay:
            Initial backoff delay in seconds; doubles each retry.

        Returns
        -------
        str
            Raw generated text from the model.

        Raises
        ------
        ClientError
            Re-raised after *max_retries* exhausted, or on non-throttling errors.
        """
        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                return self.invoke(prompt, system, model_id=model_id, max_tokens=max_tokens)
            except ClientError as exc:
                code = exc.response["Error"]["Code"]
                if code in ("ThrottlingException", "TooManyRequestsException"):
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(
                            "Throttled by Bedrock (attempt %d/%d). Retrying in %.1fs.",
                            attempt + 1, max_retries, delay,
                        )
                        time.sleep(delay)
                        last_exc = exc
                        continue
                raise
        # Should not reach here, but satisfy type checker
        raise last_exc  # type: ignore[misc]
