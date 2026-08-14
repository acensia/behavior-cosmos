"""Smoke client for the Cosmos3 Edge/Nano DROID policy server (openpi protocol).

Run INSIDE the cosmos-framework venv (it provides websockets + openpi-server):

  cosmos-framework/.venv/bin/python scripts/smoke_droid_client.py --port 8000

Sends one dummy DROID-style observation (3 black cameras + zero joints) and
checks that an action chunk of shape (32, 8) comes back — [joint_pos(7),
gripper(1)] absolute joint positions, per the Cosmos3-Policy-DROID contract.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np


def _load_msgpack_numpy():
    try:
        from openpi_client import msgpack_numpy  # type: ignore

        return msgpack_numpy
    except ModuleNotFoundError:
        from openpi.serving import msgpack_numpy  # type: ignore

        return msgpack_numpy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--timeout", type=float, default=600.0, help="seconds to wait for the server")
    parser.add_argument("--prompt", default="pick up the banana and put it in the bowl")
    args = parser.parse_args()

    import websockets.sync.client

    msgpack_numpy = _load_msgpack_numpy()
    packer = msgpack_numpy.Packer()

    uri = f"ws://{args.host}:{args.port}"
    deadline = time.monotonic() + args.timeout
    conn = None
    print(f"Connecting to {uri} (waiting up to {args.timeout:.0f}s for the server)...")
    while time.monotonic() < deadline:
        try:
            conn = websockets.sync.client.connect(uri, max_size=None, open_timeout=10)
            break
        except Exception:
            time.sleep(5.0)
    if conn is None:
        print("ERROR: server did not come up in time", file=sys.stderr)
        return 1

    with conn:
        metadata = msgpack_numpy.unpackb(conn.recv())
        print(f"Server metadata: {metadata}")

        obs = {
            "prompt": args.prompt,
            "observation/exterior_image_1_left": np.zeros((180, 320, 3), dtype=np.uint8),
            "observation/exterior_image_2_left": np.zeros((180, 320, 3), dtype=np.uint8),
            "observation/wrist_image_left": np.zeros((180, 320, 3), dtype=np.uint8),
            "observation/joint_position": np.zeros(7, dtype=np.float32),
            "observation/gripper_position": np.zeros(1, dtype=np.float32),
        }
        t0 = time.monotonic()
        conn.send(packer.pack(obs))
        response = msgpack_numpy.unpackb(conn.recv())
        dt = time.monotonic() - t0

    action = np.asarray(response.get("action"))
    print(f"Inference took {dt:.1f}s; action chunk shape: {action.shape}, dtype: {action.dtype}")
    print(f"First action: {np.array2string(action[0], precision=4)}")
    if action.ndim != 2 or action.shape[1] != 8:
        print(f"ERROR: expected (T, 8) action chunk, got {action.shape}", file=sys.stderr)
        return 1
    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
