"""AWS Bedrock clients for real-time and batch inference.

Two classes are provided:

BedrockClient
    Real-time inference using the Bedrock Converse API.  Works with any
    Converse-compatible model (Nova Micro, Nova Lite, Llama 3, Claude, etc.).

BedrockBatchClient
    Asynchronous batch inference using ``CreateModelInvocationJob``.  Requires
    an S3 bucket for input/output and an IAM role that Bedrock can assume.
    Suitable for Nova Micro batch processing at ~40× lower cost than real-time.

Credentials are sourced from the standard boto3 chain
(environment variables, ~/.aws/credentials, IAM role, etc.).
AWS_REGION environment variable controls the region; defaults to us-east-1.
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

_DEFAULT_REGION = "us-east-1"
_DEFAULT_MODEL  = "amazon.nova-micro-v1:0"

_TERMINAL_STATES = {"Completed", "Failed", "Stopped"}
_POLL_INTERVAL   = 30    # seconds between status checks


# ---------------------------------------------------------------------------
# Real-time client  (Converse API)
# ---------------------------------------------------------------------------

class BedrockClient:
    """Real-time inference via the Bedrock Converse API.

    Parameters
    ----------
    region:
        AWS region.  Reads ``AWS_REGION`` env var; defaults to us-east-1.
    model_id:
        Default model ID.  Defaults to ``amazon.nova-micro-v1:0``.
    """

    def __init__(
        self,
        region: str | None = None,
        model_id: str = _DEFAULT_MODEL,
    ) -> None:
        self.region = region or os.environ.get("AWS_REGION", _DEFAULT_REGION)
        self.default_model_id = model_id
        self._client = boto3.client("bedrock-runtime", region_name=self.region)
        logger.debug(
            "BedrockClient initialised (region=%s, model=%s)",
            self.region, self.default_model_id,
        )

    def invoke(
        self,
        system: str,
        user_message: str,
        model_id: str | None = None,
        max_tokens: int = 256,
    ) -> str:
        """Call the Converse API once (no retry).

        Returns
        -------
        str
            Text content of the assistant response.
        """
        effective_model = model_id or self.default_model_id
        response = self._client.converse(
            modelId=effective_model,
            messages=[{"role": "user", "content": [{"text": user_message}]}],
            system=[{"text": system}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": 0.0},
        )
        return response["output"]["message"]["content"][0]["text"].strip()

    def invoke_with_retry(
        self,
        system: str,
        user_message: str,
        model_id: str | None = None,
        max_tokens: int = 256,
        max_retries: int = 4,
        base_delay: float = 2.0,
        # Legacy keyword kept for call-site compatibility
        prompt: str | None = None,
    ) -> str:
        """Call Converse API with exponential backoff on throttling.

        ``prompt`` is accepted as a legacy alias for ``user_message``
        so existing call sites that pass ``prompt=`` still work.
        """
        if prompt is not None and not user_message:
            user_message = prompt

        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                return self.invoke(
                    system=system,
                    user_message=user_message,
                    model_id=model_id,
                    max_tokens=max_tokens,
                )
            except ClientError as exc:
                code = exc.response["Error"]["Code"]
                if code in ("ThrottlingException", "TooManyRequestsException"):
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(
                            "Throttled (attempt %d/%d). Retrying in %.1fs.",
                            attempt + 1, max_retries, delay,
                        )
                        time.sleep(delay)
                        last_exc = exc
                        continue
                raise
        raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Batch client
# ---------------------------------------------------------------------------

class BedrockBatchClient:
    """Batch inference via the Bedrock ``CreateModelInvocationJob`` API.

    Requires:
    - An S3 bucket accessible from the Bedrock service.
    - An IAM role ARN that Bedrock can assume, with read/write access to
      that bucket.

    Typical workflow
    ----------------
    ::

        client = BedrockBatchClient()
        records = [
            client.build_record(row_id, system_prompt, user_prompt)
            for row_id, system_prompt, user_prompt in items
        ]
        s3_input_uri = "s3://bucket/prefix/input.jsonl"
        s3_output_uri = "s3://bucket/prefix/output/"

        client.upload_input(records, s3_input_uri)
        job_arn = client.submit_job(
            job_name="my-job",
            model_id="amazon.nova-micro-v1:0",
            s3_input_uri=s3_input_uri,
            s3_output_uri=s3_output_uri,
            role_arn="arn:aws:iam::123456789:role/BedrockBatchRole",
        )
        status = client.wait_for_completion(job_arn)
        results = client.download_results(s3_output_uri)

    Parameters
    ----------
    region:
        AWS region.  Reads ``AWS_REGION`` env var; defaults to us-east-1.
    """

    def __init__(self, region: str | None = None) -> None:
        self.region = region or os.environ.get("AWS_REGION", _DEFAULT_REGION)
        self._bedrock = boto3.client("bedrock", region_name=self.region)
        self._s3      = boto3.client("s3",       region_name=self.region)
        logger.debug("BedrockBatchClient initialised (region=%s)", self.region)

    # ------------------------------------------------------------------
    # Record construction
    # ------------------------------------------------------------------

    def build_record(
        self,
        record_id: str,
        system: str,
        user_message: str,
        max_tokens: int = 256,
    ) -> dict:
        """Return one batch JSONL record dict for Nova Micro.

        The ``schemaVersion: "messages-v1"`` key is required by Bedrock batch
        for Amazon Nova models.

        Parameters
        ----------
        record_id:
            Unique identifier for this record (used to join output back to
            input rows).  Should match the candidate ``id`` field.
        system:
            System prompt text.
        user_message:
            User-turn prompt text.
        max_tokens:
            Maximum tokens to generate.
        """
        return {
            "recordId": record_id,
            "modelInput": {
                "schemaVersion": "messages-v1",
                "messages": [
                    {"role": "user", "content": [{"text": user_message}]}
                ],
                "system": [{"text": system}],
                "inferenceConfig": {
                    "maxTokens": max_tokens,
                    "temperature": 0,
                },
            },
        }

    # ------------------------------------------------------------------
    # S3 helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_s3_uri(uri: str) -> tuple[str, str]:
        """Split ``s3://bucket/key`` → ``(bucket, key)``."""
        if not uri.startswith("s3://"):
            raise ValueError(f"Expected s3:// URI, got: {uri!r}")
        parts = uri[5:].split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""
        return bucket, key

    def upload_input(self, records: list[dict], s3_uri: str) -> None:
        """Serialise *records* as JSONL and upload to *s3_uri*.

        Parameters
        ----------
        records:
            List of dicts produced by :meth:`build_record`.
        s3_uri:
            Destination S3 URI, e.g. ``s3://bucket/prefix/input.jsonl``.
        """
        bucket, key = self._parse_s3_uri(s3_uri)
        body = "\n".join(json.dumps(r, ensure_ascii=False) for r in records)
        self._s3.put_object(Bucket=bucket, Key=key, Body=body.encode("utf-8"))
        logger.info(
            "Uploaded %d records (%d KB) → %s",
            len(records), len(body) // 1024, s3_uri,
        )

    def download_results(self, s3_output_uri: str) -> list[dict]:
        """Download and parse all ``.jsonl.out`` files under *s3_output_uri*.

        Bedrock writes batch output to the prefix you specified, adding a
        sub-folder named after the job.  This method lists all ``.jsonl.out``
        objects under the prefix and concatenates them.

        Returns
        -------
        list of dict
            Each dict has keys ``recordId`` and either ``modelOutput``
            (success) or ``error`` (failure).
        """
        bucket, prefix = self._parse_s3_uri(s3_output_uri)
        paginator = self._s3.get_paginator("list_objects_v2")
        results: list[dict] = []
        files_found = 0

        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.endswith(".jsonl.out"):
                    continue
                files_found += 1
                raw = self._s3.get_object(Bucket=bucket, Key=key)["Body"].read()
                for line in raw.decode("utf-8").splitlines():
                    line = line.strip()
                    if line:
                        try:
                            results.append(json.loads(line))
                        except json.JSONDecodeError as exc:
                            logger.warning("Skipping malformed output line: %s", exc)

        logger.info(
            "Downloaded %d result records from %d output file(s) at %s",
            len(results), files_found, s3_output_uri,
        )
        return results

    # ------------------------------------------------------------------
    # Job management
    # ------------------------------------------------------------------

    def submit_job(
        self,
        job_name: str,
        model_id: str,
        s3_input_uri: str,
        s3_output_uri: str,
        role_arn: str,
    ) -> str:
        """Submit a batch inference job and return its ARN.

        Parameters
        ----------
        job_name:
            Human-readable name.  Must match ``[a-zA-Z0-9](-*[a-zA-Z0-9])*``
            and be ≤ 63 characters.
        model_id:
            Bedrock model ID, e.g. ``amazon.nova-micro-v1:0``.
        s3_input_uri:
            S3 URI of the input JSONL file.
        s3_output_uri:
            S3 URI prefix for output.  Bedrock will write into a sub-folder
            of this prefix.
        role_arn:
            ARN of the IAM role Bedrock assumes to access S3.
        """
        response = self._bedrock.create_model_invocation_job(
            jobName=job_name,
            roleArn=role_arn,
            modelId=model_id,
            inputDataConfig={
                "s3InputDataConfig": {
                    "s3Uri": s3_input_uri,
                    "s3InputFormat": "JSONL",
                }
            },
            outputDataConfig={
                "s3OutputDataConfig": {
                    "s3Uri": s3_output_uri,
                }
            },
        )
        job_arn = response["jobArn"]
        logger.info("Submitted batch job: %s  ARN: %s", job_name, job_arn)
        return job_arn

    def get_job_status(self, job_arn: str) -> tuple[str, dict]:
        """Return ``(status_string, full_response_dict)`` for *job_arn*."""
        response = self._bedrock.get_model_invocation_job(jobIdentifier=job_arn)
        return response["status"], response

    def wait_for_completion(
        self,
        job_arn: str,
        poll_interval: int = _POLL_INTERVAL,
        max_wait: int = 86_400,
    ) -> str:
        """Block until the job reaches a terminal state.

        Logs status updates on every poll.

        Parameters
        ----------
        job_arn:
            Job ARN returned by :meth:`submit_job`.
        poll_interval:
            Seconds between status checks.
        max_wait:
            Maximum total wait in seconds (default 24 h).

        Returns
        -------
        str
            Final status string: ``"Completed"``, ``"Failed"``, or
            ``"Stopped"``.

        Raises
        ------
        TimeoutError
            If *max_wait* elapses before a terminal state.
        RuntimeError
            If the job enters ``"Failed"`` status.
        """
        started = time.monotonic()
        while True:
            status, info = self.get_job_status(job_arn)
            elapsed = int(time.monotonic() - started)
            logger.info("Batch job status: %s  (elapsed %ds)", status, elapsed)

            if status in _TERMINAL_STATES:
                if status == "Failed":
                    msg = info.get("failureMessage", "no details available")
                    raise RuntimeError(f"Batch job failed: {msg}")
                return status

            if elapsed >= max_wait:
                raise TimeoutError(
                    f"Batch job did not complete within {max_wait}s. "
                    f"Last status: {status}. ARN: {job_arn}"
                )
            time.sleep(poll_interval)
