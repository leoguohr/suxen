"""Package completed CAD50 terminal-FFN evidence; never runs training or inference."""

import argparse
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STEPS = (0, 25, 50, 75, 100)
UIDS = [f"teacher_cad50_{i:02d}" for i in range(50)]
EXCLUDED_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".zip", ".npy", ".npz", ".tmp", ".lock", ".pyc"}
TEXT_SUFFIXES = {".py", ".json", ".jsonl", ".md", ".txt", ".log", ".csv", ".patch", ".toml", ".yaml", ".yml"}
SECRET_PATTERNS = (
    re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    re.compile(rb"sk-[A-Za-z0-9]{20,}"),
    re.compile(rb"(?i)(?:password|api[_-]?key|secret)\s*[=:]\s*['\"][^'\"]{4,}"),
)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def safe_relative(path):
    value = Path(path)
    if value.is_absolute() or not value.parts or ".." in value.parts:
        raise ValueError(f"Unsafe relative path: {path}")
    return value


def copy_small_tree(source, destination):
    if not source.is_dir():
        raise FileNotFoundError(source)
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if path.is_symlink() or {".git", "__pycache__", "_rigorpilot"} & set(relative.parts):
            continue
        if path.suffix.lower() in EXCLUDED_SUFFIXES or path.name == ".DS_Store":
            continue
        if path.stat().st_size > 20 * 1024 * 1024:
            raise ValueError(f"Unexpected large evidence file: {path}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)


def checked_predictions(source, destination):
    """Copy five full evaluations and verify each of the 250 saved arrays."""
    completed = read_json(source / "complete.json")
    if (completed["state"], completed["new_updates"], completed["completed_updates"]) != ("complete", 100, 2600):
        raise ValueError(f"Branch has not completed 100 updates: {source}")
    expected = set()
    checkpoint_hashes = []
    for step in STEPS:
        name = f"eval-new{step:04d}.json"
        evaluation = read_json(source / name)
        if evaluation["new_step"] != step or evaluation["completed_updates"] != 2500 + step:
            raise ValueError(f"Wrong checkpoint identity: {source / name}")
        rows = evaluation["meshes"]
        if [row["uid"] for row in rows] != UIDS:
            raise ValueError(f"Incomplete or reordered CAD50 evaluation: {source / name}")
        checkpoint_hashes.append(evaluation["checkpoint_sha256"])
        for row in rows:
            relative = safe_relative(row["prediction_path"])
            if relative != Path(f"predictions-new{step:04d}") / f"{row['uid']}.npz":
                raise ValueError(f"Unexpected prediction path: {relative}")
            path = source / relative
            if not path.is_file() or path.is_symlink() or sha(path) != row["prediction_sha256"]:
                raise ValueError(f"Missing or changed prediction: {path}")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            expected.add(relative)
    actual = {path.relative_to(source) for path in source.glob("predictions-new*/*.npz")}
    if actual != expected or len(expected) != 250:
        raise ValueError(f"Expected exactly 250 predictions in {source}; found {len(actual)}")
    return checkpoint_hashes


def checked_cold_verify(source, destination, final_checkpoint_sha256):
    cold = read_json(ROOT / "repro_outputs/COLD_VERIFY.json")
    if (cold.get("passed"), cold.get("optimizer_updates"),
            cold.get("all50_arrays_bitwise_equal")) != (True, 0, True):
        raise ValueError("Cold verification did not pass with zero updates and equal arrays")
    evaluation = read_json(source / "cold_verify/eval-new0100.json")
    if (evaluation["new_step"], evaluation["completed_updates"],
            evaluation["checkpoint_sha256"]) != (100, 2600, final_checkpoint_sha256):
        raise ValueError("Cold evaluation checkpoint does not match final Treatment evaluation")
    if cold["checkpoint_sha256"] != final_checkpoint_sha256:
        raise ValueError("Cold verification checkpoint hash differs from Treatment")
    rows = evaluation["meshes"]
    checks = cold["meshes"]
    if [row["uid"] for row in rows] != UIDS or [row["uid"] for row in checks] != UIDS:
        raise ValueError("Cold verification is missing CAD50 meshes")
    expected = set()
    for row, check in zip(rows, checks):
        relative = safe_relative(row["prediction_path"])
        if relative != Path("cold_verify/predictions-new0100") / f"{row['uid']}.npz":
            raise ValueError(f"Unexpected cold prediction path: {relative}")
        path = source / relative
        if (check.get("all_arrays_bitwise_equal") is not True or
                check["prediction_sha256"] != row["prediction_sha256"] or
                not path.is_file() or path.is_symlink() or sha(path) != row["prediction_sha256"]):
            raise ValueError(f"Missing or changed cold prediction: {path}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        expected.add(relative)
    actual = {path.relative_to(source) for path in source.glob("cold_verify/predictions-new0100/*.npz")}
    if actual != expected or len(expected) != 50:
        raise ValueError(f"Expected exactly 50 cold predictions; found {len(actual)}")


def write_baseline_diff(destination):
    before = ROOT / "baseline_source"
    after = ROOT / "H_terminal_ffn"
    baseline_names = {p.name for p in before.iterdir() if p.is_file() and p.suffix in {".py", ".json"}}
    new_python = {p.name for p in after.iterdir() if p.is_file() and p.suffix == ".py"}
    names = sorted(baseline_names | new_python)
    lines = []
    for name in names:
        old, new = before / name, after / name
        a = old.read_text().splitlines(keepends=True) if old.exists() else []
        b = new.read_text().splitlines(keepends=True) if new.exists() else []
        lines.extend(difflib.unified_diff(a, b, fromfile=f"baseline_source/{name}" if old.exists() else "/dev/null",
                                         tofile=f"H_terminal_ffn/{name}" if new.exists() else "/dev/null"))
    (destination / "DIFF_FROM_BASELINE.patch").write_text("".join(lines))


def git_evidence(destination):
    def command(*args):
        return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout

    commit = command("rev-parse", "HEAD").strip()
    (destination / "GIT_HEAD.txt").write_text(commit + "\n")
    (destination / "GIT_STATUS.txt").write_text(command("status", "--short"))
    (destination / "GIT_LOG.txt").write_text(command("log", "-5", "--format=%H %s"))
    (destination / "WORKTREE_DIFF.patch").write_text(command("diff", "--no-ext-diff", "HEAD"))
    (destination / "INDEX_DIFF.patch").write_text(command("diff", "--no-ext-diff", "--cached"))
    bundle = destination / "experiment.git.bundle"
    subprocess.run(["git", "bundle", "create", str(bundle), "--all"], cwd=ROOT, check=True, capture_output=True)
    if bundle.stat().st_size == 0:
        raise ValueError("Empty Git bundle")
    return commit


def scan_and_manifest(bundle):
    records = []
    for path in sorted(bundle.rglob("*")):
        if not path.is_file() or path == bundle / "FILE_SHA256.json":
            continue
        if path.is_symlink():
            raise ValueError(f"Symlink in package: {path}")
        relative = path.relative_to(bundle).as_posix()
        if path.suffix.lower() in {".pt", ".pth", ".ckpt", ".safetensors", ".zip"}:
            raise ValueError(f"Large model/archive in package: {relative}")
        if path.suffix.lower() in TEXT_SUFFIXES:
            data = path.read_bytes()
            if any(pattern.search(data) for pattern in SECRET_PATTERNS):
                raise ValueError(f"Potential credential in package: {relative}")
        records.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha(path)})
    (bundle / "FILE_SHA256.json").write_text(json.dumps({"files": records}, indent=2) + "\n")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path.home() / "Downloads/CAD50_Terminal_FFN_Pair_20260923")
    args = parser.parse_args()
    h = ROOT / "H_terminal_ffn"
    control = ROOT / "Control_reused"
    recovered = ROOT.parent / "teacher_original_zip_reaudit_20260923"
    structure = ROOT.parent / "original512_structure_diagnosis_20260923"
    final_report = ROOT / "repro_outputs/REPORT.md"
    checkpoint_manifest = ROOT / "repro_outputs/CHECKPOINT_MANIFEST.json"
    for required in [final_report, checkpoint_manifest, ROOT / "repro_outputs/COLD_VERIFY.json",
                     h / "cold_verify/eval-new0100.json", control / "complete.json", h / "updates.jsonl"]:
        if not required.is_file():
            raise FileNotFoundError(f"Final evidence is not ready: {required}")
    if len((h / "updates.jsonl").read_text().splitlines()) != 100:
        raise ValueError("Treatment update log is incomplete")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_zip = args.output_dir / "CAD50_Terminal_FFN_Pair_20260923_Evidence.zip"
    if final_zip.exists():
        raise FileExistsError(f"Sealed delivery already exists: {final_zip}")
    with tempfile.TemporaryDirectory(prefix="package-", dir=args.output_dir) as temporary:
        work = Path(temporary)
        bundle = work / "CAD50_Terminal_FFN_Pair_20260923"
        bundle.mkdir()
        copy_small_tree(ROOT, bundle / "experiment")
        copy_small_tree(recovered, bundle / "original_zip_recovery")
        copy_small_tree(structure, bundle / "original512_structure")
        h_hashes = checked_predictions(h, bundle / "experiment/H_terminal_ffn")
        checked_predictions(control, bundle / "experiment/Control_reused")
        checked_cold_verify(h, bundle / "experiment/H_terminal_ffn", h_hashes[-1])
        checkpoint_text = checkpoint_manifest.read_text()
        if not all(value in checkpoint_text for value in h_hashes):
            raise ValueError("Checkpoint manifest does not cover all five Treatment evaluations")
        write_baseline_diff(bundle)
        commit = git_evidence(bundle)
        (bundle / "PACKAGE_INDEX.md").write_text(
            "# CAD50 末端FFN对照证据\n\n"
            "实验结论见 `experiment/repro_outputs/REPORT.md`；本脚本只核对文件完整性。"
            "Treatment 与复用的 Control 各含 0/25/50/75/100 五次全50预测，"
            "另含 Treatment 最终冷加载验证的50份预测。"
            "完整 checkpoint 路径、大小和 SHA256 见 `experiment/repro_outputs/CHECKPOINT_MANIFEST.json`。"
            "`original_zip_recovery/` 含原ZIP本次逆向代码与证据；"
            "`original512_structure/` 含原512结构诊断。"
            "大权重、原始ZIP、Adam张量与原数据均未打包。\n\n"
            f"Git commit: `{commit}`。`experiment.git.bundle` 可复查已提交代码；"
            "`DIFF_FROM_BASELINE.patch` 展示本轮分支相对 baseline_source 的代码差异。\n"
        )
        records = scan_and_manifest(bundle)
        temporary_zip = work / final_zip.name
        with zipfile.ZipFile(temporary_zip, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=6, allowZip64=True) as archive:
            for path in sorted(bundle.rglob("*")):
                if path.is_file():
                    archive.write(path, arcname=path.relative_to(work).as_posix())
        with zipfile.ZipFile(temporary_zip) as archive:
            if archive.testzip() is not None:
                raise ValueError("ZIP CRC verification failed")
            prefix = bundle.name + "/"
            for record in records:
                if hashlib.sha256(archive.read(prefix + record["path"])).hexdigest() != record["sha256"]:
                    raise ValueError(f"ZIP content hash mismatch: {record['path']}")
            if len(archive.namelist()) != len(records) + 1:
                raise ValueError("ZIP member count mismatch")
        receipt = {"zip": str(final_zip), "bytes": temporary_zip.stat().st_size,
                   "sha256": sha(temporary_zip), "files": len(records) + 1,
                   "file_manifest_sha256": sha(bundle / "FILE_SHA256.json"),
                   "zip_crc": "passed", "all_file_hashes": "passed", "git_commit": commit}
        os.replace(temporary_zip, final_zip)
        final_zip.chmod(0o444)
        (args.output_dir / "delivery.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt))


if __name__ == "__main__":
    main()
