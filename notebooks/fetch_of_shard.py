"""Resumable download of an OrthoFoundation shard's kernel output.

`kaggle kernels output` is not resumable and Kaggle breaks the connection part-way through a
551-file shard (BrokenPipeError, three times now). Worse, the CLI opens a request for EVERY file
before deciding it already has it, so a retry re-pays for the whole prefix and only makes progress
because the listing order shifts between attempts -- 68, then 111, then 209 files over three runs.

This pages the listing itself, skips any file already on disk in the staging dir OR already moved
into the cache, and retries per page. The kernel writes _shardN.json last, so its arrival is the
completion signal; a file count is not, because shard 7 legitimately holds 550 and the rest 551.
"""
import os, sys, time
from pathlib import Path
import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

SHARD = int(sys.argv[1])
STAGE = Path(f"data/_of_staging/s{SHARD}/features"); STAGE.mkdir(parents=True, exist_ok=True)
CACHE = Path("data/_of_features")
KERNEL = f"rsna-knee-of-screen-s{SHARD}"

api = KaggleApi(); api.authenticate()

def have(name: str) -> bool:
    base = os.path.basename(name)
    return (STAGE / base).exists() or (CACHE / base).exists()

def pages():
    """Yield (file_name, url) across every page, retrying a page whose listing call fails."""
    token, seen = None, 0
    while True:
        for attempt in range(6):
            try:
                with api.build_kaggle_client() as client:
                    req = ApiListKernelSessionOutputRequest()
                    req.user_name, req.kernel_slug = "raahimnawaz", KERNEL
                    api._set_paging(req, 200, token)
                    resp = client.kernels.kernels_api_client.list_kernel_session_output(req)
                break
            except Exception as e:
                if attempt == 5: raise
                print(f"  listing retry {attempt+1}: {type(e).__name__}", flush=True); time.sleep(3)
        for item in resp.files or []:
            seen += 1
            yield item.file_name, item.url
        token = resp.next_page_token
        if not token:
            print(f"  listing exhausted at {seen} files", flush=True); return

got = skipped = 0
for name, url in pages():
    if have(name):
        skipped += 1; continue
    for attempt in range(6):
        try:
            r = requests.get(url, timeout=120); r.raise_for_status()
            (STAGE / os.path.basename(name)).write_bytes(r.content)
            got += 1
            break
        except Exception as e:
            if attempt == 5:
                print(f"  GAVE UP on {name}: {type(e).__name__}", flush=True); break
            time.sleep(2 * (attempt + 1))
    if got and got % 50 == 0:
        print(f"  {got} fetched, {skipped} already held", flush=True)

print(f"s{SHARD}: fetched {got}, skipped {skipped}, npz now {len(list(STAGE.glob('*.npz')))}, "
      f"manifest {'YES' if (STAGE/f'_shard{SHARD}.json').exists() else 'NO'}", flush=True)
