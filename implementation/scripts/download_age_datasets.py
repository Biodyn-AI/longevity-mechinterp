#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import requests

DEFAULT_CHUNK_SIZE = 8 * 1024 * 1024


@dataclass
class DatasetSpec:
    dataset_id: str
    phase: str
    filename: str
    expected_bytes: int
    url: str
    source: str
    notes: str


@dataclass
class DownloadResult:
    dataset_id: str
    status: str
    bytes_downloaded: int
    expected_bytes: int
    output_path: Path
    message: str


def _load_manifest(path: Path) -> List[DatasetSpec]:
    specs: List[DatasetSpec] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            specs.append(
                DatasetSpec(
                    dataset_id=row["dataset_id"],
                    phase=row["phase"],
                    filename=row["filename"],
                    expected_bytes=int(row["bytes"]),
                    url=row["url"],
                    source=row.get("source", ""),
                    notes=row.get("notes", ""),
                )
            )
    return specs


def _format_bytes(num_bytes: int) -> str:
    gib = num_bytes / (1024**3)
    return f"{num_bytes} ({gib:.2f} GiB)"


def _supports_range(url: str, timeout_s: int) -> bool:
    try:
        resp = requests.get(url, headers={"Range": "bytes=0-0"}, stream=True, timeout=timeout_s)
        status = resp.status_code
        resp.close()
        return status == 206
    except Exception:
        return False


def _free_bytes(path: Path) -> int:
    usage = shutil.disk_usage(path)
    return int(usage.free)


def _download_one(
    spec: DatasetSpec,
    output_dir: Path,
    timeout_s: int,
    verify_ssl: bool,
    dry_run: bool,
) -> DownloadResult:
    output_path = output_dir / spec.filename
    part_path = output_dir / f"{spec.filename}.part"

    # Already complete: skip.
    if output_path.exists() and output_path.stat().st_size == spec.expected_bytes:
        return DownloadResult(
            dataset_id=spec.dataset_id,
            status="already_present",
            bytes_downloaded=spec.expected_bytes,
            expected_bytes=spec.expected_bytes,
            output_path=output_path,
            message="file already exists with expected size",
        )

    resume_ok = _supports_range(spec.url, timeout_s=timeout_s)

    existing_part = part_path.stat().st_size if part_path.exists() else 0
    if existing_part > 0 and not resume_ok:
        # If server does not support ranges, stale partials cannot be resumed.
        part_path.unlink()
        existing_part = 0

    if dry_run:
        mode = "resume" if (resume_ok and existing_part > 0) else "fresh"
        return DownloadResult(
            dataset_id=spec.dataset_id,
            status="dry_run",
            bytes_downloaded=existing_part,
            expected_bytes=spec.expected_bytes,
            output_path=output_path,
            message=f"would run {mode} download (range_supported={resume_ok})",
        )

    headers = {}
    mode = "wb"
    if resume_ok and existing_part > 0:
        headers["Range"] = f"bytes={existing_part}-"
        mode = "ab"

    with requests.get(spec.url, headers=headers, stream=True, timeout=timeout_s, verify=verify_ssl) as resp:
        if resp.status_code not in (200, 206):
            return DownloadResult(
                dataset_id=spec.dataset_id,
                status="failed",
                bytes_downloaded=existing_part,
                expected_bytes=spec.expected_bytes,
                output_path=output_path,
                message=f"HTTP {resp.status_code}",
            )

        downloaded = existing_part
        last_print = time.time()

        with part_path.open(mode) as out:
            for chunk in resp.iter_content(chunk_size=DEFAULT_CHUNK_SIZE):
                if not chunk:
                    continue
                out.write(chunk)
                downloaded += len(chunk)

                now = time.time()
                if now - last_print >= 15:
                    pct = 100.0 * downloaded / spec.expected_bytes if spec.expected_bytes > 0 else 0.0
                    print(
                        f"[{spec.dataset_id}] {pct:6.2f}%  {downloaded}/{spec.expected_bytes} bytes",
                        flush=True,
                    )
                    last_print = now

    actual_size = part_path.stat().st_size
    if actual_size != spec.expected_bytes:
        return DownloadResult(
            dataset_id=spec.dataset_id,
            status="failed",
            bytes_downloaded=actual_size,
            expected_bytes=spec.expected_bytes,
            output_path=output_path,
            message=f"size mismatch after download: {actual_size} vs expected {spec.expected_bytes}",
        )

    part_path.rename(output_path)
    return DownloadResult(
        dataset_id=spec.dataset_id,
        status="downloaded",
        bytes_downloaded=actual_size,
        expected_bytes=spec.expected_bytes,
        output_path=output_path,
        message="ok",
    )


def _write_log(log_path: Path, rows: Iterable[DownloadResult]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset_id", "status", "bytes_downloaded", "expected_bytes", "output_path", "message"])
        for row in rows:
            writer.writerow(
                [
                    row.dataset_id,
                    row.status,
                    row.bytes_downloaded,
                    row.expected_bytes,
                    str(row.output_path),
                    row.message,
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download age-diverse single-cell datasets from manifest.")
    parser.add_argument(
        "--manifest",
        default=str(
            Path(__file__).resolve().parents[1]
            / "data_downloads"
            / "manifests"
            / "age_dataset_manifest.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1] / "data_downloads" / "raw"),
    )
    parser.add_argument("--phase", default="core", help="Manifest phase to download (e.g., core, extended, all)")
    parser.add_argument("--dataset-id", action="append", default=None, help="Optional specific dataset IDs to download")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--insecure", action="store_true", help="Disable SSL verification")
    parser.add_argument(
        "--log-path",
        default=str(Path(__file__).resolve().parents[1] / "data_downloads" / "logs" / "download_log.csv"),
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = _load_manifest(manifest_path)

    if args.phase != "all":
        specs = [s for s in specs if s.phase == args.phase]

    if args.dataset_id:
        wanted = set(args.dataset_id)
        specs = [s for s in specs if s.dataset_id in wanted]

    if not specs:
        raise ValueError("No datasets selected from manifest. Check --phase/--dataset-id.")

    total_expected = sum(s.expected_bytes for s in specs)
    free_now = _free_bytes(output_dir)

    print(f"Selected datasets: {len(specs)}")
    print(f"Expected total: {_format_bytes(total_expected)}")
    print(f"Free space: {_format_bytes(free_now)}")

    # Keep a 20 GiB buffer to avoid filling the volume.
    safety_buffer = 20 * 1024**3
    if not args.dry_run and free_now < total_expected + safety_buffer:
        raise RuntimeError(
            "Insufficient free space for selected downloads with safety buffer. "
            f"Need {_format_bytes(total_expected + safety_buffer)}, have {_format_bytes(free_now)}"
        )

    results: List[DownloadResult] = []
    for spec in specs:
        print(f"\n=== {spec.dataset_id} ({spec.source}) ===")
        print(f"File: {spec.filename}")
        print(f"Expected: {_format_bytes(spec.expected_bytes)}")
        print(f"URL: {spec.url}")
        result = _download_one(
            spec=spec,
            output_dir=output_dir,
            timeout_s=args.timeout,
            verify_ssl=not args.insecure,
            dry_run=args.dry_run,
        )
        print(f"Result: {result.status} | {result.message}")
        results.append(result)

    log_path = Path(args.log_path).resolve()
    _write_log(log_path, results)
    print(f"\nLog written: {log_path}")

    failed = [r for r in results if r.status == "failed"]
    if failed:
        print("\nFailures:")
        for item in failed:
            print(f"- {item.dataset_id}: {item.message}")
        sys.exit(2)


if __name__ == "__main__":
    main()
