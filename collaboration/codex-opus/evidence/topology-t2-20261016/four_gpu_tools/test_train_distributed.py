"""CPU-only, bounded 50-object checkpoint/resume acceptance for the independent adapter."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import socket

import torch


def load(output):
    pointer = json.loads((output / "latest.json").read_text())
    path = output / pointer["file"]
    assert path.stat().st_size == pointer["bytes"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == pointer["sha256"]
    return torch.load(path, map_location="cpu", weights_only=False)


def equal(a, b, path="root"):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b), path
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for key in a:
            equal(a[key], b[key], path + "/" + str(key))
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for index, (x, y) in enumerate(zip(a, b)):
            equal(x, y, path + "/" + str(index))
    else:
        import numpy as np
        if isinstance(a, np.ndarray):
            assert np.array_equal(a, b), path
        else:
            assert a == b, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--world", type=int, default=2)
    parser.add_argument("--master-port", type=int, default=None)
    args = parser.parse_args()
    trainer = Path(__file__).with_name("train_distributed.py")
    base = [sys.executable, "-m", "torch.distributed.run", "--master-addr", "127.0.0.1",
            f"--nproc_per_node={args.world}",
            str(trainer), "--source", args.source, "--tiny", "--device", "cpu", "--attention", "sdpa",
            "--eval-every", "0", "--probe-every", "0", "--final-weights", "", "--threads", "1"]

    def run(folder, steps, extra=(), success=True):
        port = args.master_port
        if port is None:
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
        command = base.copy()
        command[3:3] = ["--master-port", str(port)]
        result = subprocess.run(command + ["--output", str(folder), "--stop-after-updates", str(steps), *extra])
        assert (result.returncode == 0) == success, result.returncode

    straight, resumed = args.output / "straight", args.output / "resumed"
    run(straight, 4)
    run(resumed, 2)
    before = load(resumed)
    assert before["update"] == 2 and before["args"]["max_updates"] == 10000
    run(resumed, 2)
    a, b = load(straight), load(resumed)
    assert a["update"] == b["update"] == 4
    for key in ("model", "optimizer", "ema", "distributed"):
        equal(a[key], b[key], key)
    assert len(list(resumed.glob("checkpoint-u*.pt"))) == 1
    # Altering a scientific recipe must be rejected before loading/stepping model weights.
    run(resumed, 1, ("--lr", "0.002"), success=False)
    assert load(resumed)["update"] == 4
    # A STOP marker must prevent even one additional update.
    (resumed / "STOP").write_text("CPU smoke stop\n")
    run(resumed, 1)
    assert load(resumed)["update"] == 4
    (resumed / "STOP").unlink()
    # Pointer corruption must fail before use, preserving the original checkpoint file.
    pointer_path = resumed / "latest.json"
    original = pointer_path.read_bytes()
    pointer = json.loads(original)
    pointer["sha256"] = "0" * 64
    pointer_path.write_text(json.dumps(pointer))
    run(resumed, 1, success=False)
    pointer_path.write_bytes(original)
    assert load(resumed)["update"] == 4
    # Complete-output protection, independently of the bounded smoke path.
    finished = args.output / "finished"
    run(finished, 0, ("--max-updates", "2"))
    final_before = (finished / "final.json").read_bytes()
    run(finished, 0, ("--max-updates", "2"))
    assert (finished / "final.json").read_bytes() == final_before
    # Synthetic early-pass fixture: interrupt after final-stage save and before completion marker.
    # This tests metadata continuity only; no evaluator score is fabricated or claimed.
    fixture = load(finished)
    fixture["distributed"]["passed_early"] = True
    pointer = json.loads((finished / "latest.json").read_text())
    fixture_path = finished / pointer["file"]
    torch.save(fixture, fixture_path)
    pointer.update(bytes=fixture_path.stat().st_size,
                   sha256=hashlib.sha256(fixture_path.read_bytes()).hexdigest())
    (finished / "latest.json").write_text(json.dumps(pointer))
    (finished / "final.json").unlink()
    (finished / "status.json").write_text(json.dumps(dict(state="synthetic_interrupted_final_eval")))
    run(finished, 0, ("--max-updates", "2"))
    assert json.loads((finished / "status.json").read_text())["passed_early"] is True
    slow = args.output / "speed_stop"
    run(slow, 1, ("--max-step-seconds", "0.000001"), success=False)
    assert json.loads((slow / "status.json").read_text())["state"] == "speed_limit_exceeded"
    assert load(slow)["update"] == 1
    print(f"DISTRIBUTED_RESUME_GATE_OK world={args.world} exact_model_adam_ema_rng_cursor=1 "
          "recipe_rejection=1 sha_rejection=1 stop_marker=1 completed_guard=1 "
          "early_pass_resume=1 speed_stop_saved=1")


if __name__ == "__main__":
    main()
