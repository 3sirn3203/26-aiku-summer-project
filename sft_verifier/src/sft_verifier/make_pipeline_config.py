from __future__ import annotations

import argparse
import copy
from pathlib import Path

from .common import read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an agent config for a merged verifier model")
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--merged-verifier", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model_dir = args.merged_verifier.resolve()
    if not model_dir.is_dir():
        parser.error("--merged-verifier must be an existing directory")
    payload = copy.deepcopy(read_json(args.base_config))
    verifier = payload["roles"]["verifier"]
    verifier["model"].update(
        {
            "source": "local",
            "id": str(model_dir),
            "revision": "local",
            "dtype": "float32",
            "device": "cuda:0",
            "attention_implementation": "eager",
            "trust_remote_code": False,
            "cache_dir": None,
        }
    )
    write_json(args.output, payload)
    print("wrote %s" % args.output)


if __name__ == "__main__":
    main()

