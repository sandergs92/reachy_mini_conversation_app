"""Nothing (for ruff)."""

import os


alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
if "expandable_segments" not in alloc_conf:
    if len(alloc_conf) > 0:
        alloc_conf += ",expandable_segments:True"
    else:
        alloc_conf = "expandable_segments:True"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = alloc_conf
