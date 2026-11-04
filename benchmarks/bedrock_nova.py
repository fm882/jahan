"""Benchmark-only Nova transport; explicit opt-in, AWS CLI credentials, no SDK dependency.

Nova accepts one synchronous text input. Preserve independent inputs by invoking
each separately under a bounded semaphore, never by concatenating persona texts.
"""

import asyncio
import json
from pathlib import Path
import tempfile


MODEL = "amazon.nova-2-multimodal-embeddings-v1:0"


async def embeddings(texts, *, region, concurrency=16):
    ceiling = asyncio.Semaphore(concurrency)

    async def one(text):
        async with ceiling:
            with tempfile.TemporaryDirectory(prefix="simcore-nova-") as directory:
                request = Path(directory) / "request.json"
                response = Path(directory) / "response.json"
                request.write_text(json.dumps({
                    "taskType": "SINGLE_EMBEDDING", "singleEmbeddingParams": {
                        "embeddingPurpose": "GENERIC_INDEX", "embeddingDimension": 256,
                        "text": {"truncationMode": "NONE", "value": text},
                    },
                }))
                process = await asyncio.create_subprocess_exec(
                    "aws", "bedrock-runtime", "invoke-model", "--region", region,
                    "--model-id", MODEL, "--body", f"fileb://{request}",
                    "--content-type", "application/json", "--accept", "application/json",
                    "--cli-connect-timeout", "10", "--cli-read-timeout", "60",
                    "--no-cli-pager", str(response),
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                )
                try:
                    _, stderr = await asyncio.wait_for(process.communicate(), timeout=90)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.communicate()
                    raise RuntimeError("Nova CLI invocation exceeded 90 seconds") from None
                if process.returncode:
                    # Do not persist AWS account identifiers or credential diagnostics.
                    category = "throttled" if b"Throttl" in stderr else "failed"
                    raise RuntimeError(f"Nova CLI invocation {category} (exit {process.returncode})")
                payload = json.loads(response.read_text())
                vectors = payload["embeddings"]
                if len(vectors) != 1 or vectors[0]["embeddingType"] != "TEXT" or "truncatedCharLength" in vectors[0]:
                    raise ValueError("Nova must return one untruncated text embedding")
                return vectors[0]["embedding"]

    results = await asyncio.gather(*(one(text) for text in texts), return_exceptions=True)
    error = next((result for result in results if isinstance(result, BaseException)), None)
    if error is not None:
        raise error
    return results
