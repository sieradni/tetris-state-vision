"""Read native oneDNN cache capacity inside a bounded CPU Job; no model runs."""
import argparse
import ctypes
import json
import os
from pathlib import Path

from probe_model import require_bounded_job


def main():
    ceiling = require_bounded_job()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity", type=int)
    parser.add_argument("--exercise", action="store_true")
    args = parser.parse_args()
    if args.capacity is not None:
        if not 1 <= args.capacity <= 64:
            parser.error("experimental finite cache capacity1..64 required")
        os.environ["ONEDNN_PRIMITIVE_CACHE_CAPACITY"] = str(args.capacity)
        os.environ["DNNL_PRIMITIVE_CACHE_CAPACITY"] = str(args.capacity)
    import torch
    record = {"torch": str(torch.__version__), "job_ceiling_bytes": ceiling,
              "mkldnn_available": torch.backends.mkldnn.is_available(),
              "mkldnn_enabled": torch.backends.mkldnn.enabled,
              "environment": {name: os.environ.get(name) for name in
                              ("ONEDNN_PRIMITIVE_CACHE_CAPACITY", "DNNL_PRIMITIVE_CACHE_CAPACITY")}}
    library = ctypes.CDLL(str(Path(torch.__file__).parent/"lib/torch_cpu.dll"))
    try:
        query = library.dnnl_get_primitive_cache_capacity
    except AttributeError:
        record.update(native_query_available=False, capacity=None)
    else:
        query.argtypes = [ctypes.POINTER(ctypes.c_int)]
        query.restype = ctypes.c_int
        capacity = ctypes.c_int(-1)
        status = query(ctypes.byref(capacity))
        if status != 0:
            raise RuntimeError(f"oneDNN cache query failed: {status}")
        record.update(native_query_available=True, capacity=capacity.value)
    if args.exercise:
        heights = [80, 80, *range(81, 97), 80]
        torch.set_num_threads(1)
        with torch.backends.mkldnn.verbose(torch.backends.mkldnn.VERBOSE_ON_CREATION):
            for index, height in enumerate(heights):
                print(json.dumps({"case": index, "height": height}), flush=True)
                pixels = torch.zeros(1, 16, height, 80).to_mkldnn()
                result = torch.nn.functional.conv2d(pixels, torch.ones(16, 16, 3, 3), padding=1).to_dense()
                if not bool((result == 0).all()):
                    raise RuntimeError("native cache exercise changes zero convolution")
        record["finite_convolutions"] = len(heights)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(record, output, indent=2)
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
