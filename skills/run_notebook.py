import os
import json
import base64
import time
import nbformat
import papermill as pm
from pathlib import Path
from typing import Optional
import subprocess
import shutil


def _resolve_notebook_path(notebook_path: str) -> Path:
    """
    Resolve the notebook path flexibly:
    - If absolute, return as-is.
    - If present relative to the repository root, use that.
    - Otherwise, search the repository for a matching filename.
    - Finally fall back to cwd-relative resolution.
    """
    repo_root = Path(__file__).resolve().parents[1]
    candidate = Path(notebook_path)
    if candidate.is_absolute():
        return candidate

    candidate = repo_root / notebook_path
    if candidate.exists():
        return candidate.resolve()

    basename = os.path.basename(notebook_path)
    for match in repo_root.rglob(basename):
        if match.is_file():
            return match.resolve()

    return (Path.cwd() / notebook_path).resolve()


def _collect_usage_metrics() -> dict:
    """Collect simple CPU and GPU usage metrics.

    - `cpu_percent`: system CPU utilization percentage (float) or None.
    - `gpus`: list of GPU dicts or None. Each GPU dict may contain:
        index, name, utilization_percent, memory_total_MB, memory_used_MB
    """
    metrics: dict = {"cpu_percent": None, "gpus": None}

    # CPU: try psutil if available
    try:
        import psutil

        try:
            metrics["cpu_percent"] = float(psutil.cpu_percent(interval=0.1))
        except Exception:
            metrics["cpu_percent"] = None
    except Exception:
        metrics["cpu_percent"] = None

    gpus_list: list = []

    # Try nvidia-smi first (common on systems with NVIDIA GPUs)
    try:
        if shutil.which("nvidia-smi"):
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,utilization.gpu,memory.total,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            for line in out.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 5:
                    try:
                        idx = int(parts[0])
                        name = parts[1]
                        util = float(parts[2])
                        mem_tot = int(parts[3])
                        mem_used = int(parts[4])
                        gpus_list.append(
                            {
                                "index": idx,
                                "name": name,
                                "utilization_percent": util,
                                "memory_total_MB": mem_tot,
                                "memory_used_MB": mem_used,
                            }
                        )
                    except Exception:
                        continue
    except Exception:
        pass

    # Fallback to GPUtil if available (import dynamically to avoid static linter warnings)
    if not gpus_list:
        try:
            import importlib

            GPUtil = importlib.import_module("GPUtil")

            for g in GPUtil.getGPUs():
                try:
                    gpus_list.append(
                        {
                            "id": int(g.id),
                            "name": g.name,
                            "load_percent": float(g.load * 100),
                            "memory_total_MB": int(g.memoryTotal),
                            "memory_used_MB": int(g.memoryUsed),
                        }
                    )
                except Exception:
                    continue
        except Exception:
            pass

    metrics["gpus"] = gpus_list if gpus_list else None
    return metrics


def _base_dir(nb_path: Optional[Path] = None) -> str:
    """
    Return the parent of the notebook's parent folder as the base directory.
    If `nb_path` is None, fall back to the repository root (two levels up from this file).
    """
    repo_root = Path(__file__).resolve().parents[1]
    if nb_path is None:
        return str(repo_root)
    # notebook -> parent (folder containing notebook) -> parent of that folder
    parent_of_parent = nb_path.parent.parent
    if parent_of_parent.exists():
        return str(parent_of_parent.resolve())
    return str(repo_root)


def run_notebook(notebook_path: str) -> str:
    """
    Execute a notebook via papermill and store outputs.
    Returns a JSON string with execution details.
    """
    nb_path = _resolve_notebook_path(notebook_path)

    if not nb_path.exists():
        result = {
            "success": False,
            "error": f"Notebook not found: {nb_path}",
            "output_dir": None,
            "executed_notebook_path": None,
            "summary_json_path": None,
            "execution_time_seconds": None,
            "image_paths": []
        }
        return json.dumps(result, indent=2)

    base = _base_dir(nb_path)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(base, "results", timestamp)
    os.makedirs(output_dir, exist_ok=True)

    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    executed_nb_path = os.path.join(output_dir, "executed.ipynb")

    params = {}
    params["results_dir"] = output_dir
    timeout = 600

    start_time = time.time()
    success = True
    error_message = None

    try:
        # Run from the notebook folder so relative writes go to expected paths.
        notebook_dir = str(nb_path.parent)
        old_cwd = os.getcwd()
        try:
            os.chdir(notebook_dir)
            pm.execute_notebook(
                str(nb_path),
                executed_nb_path,
                parameters=params,
                execution_timeout=timeout,
                kernel_name="python3"
            )
        finally:
            os.chdir(old_cwd)
    except Exception as e:
        success = False
        error_message = str(e)

    execution_duration = time.time() - start_time

    summary = {
        "notebook_path": str(nb_path),
        "executed_notebook_path": os.path.abspath(executed_nb_path),
        "success": success,
        "error": error_message,
        "execution_time_seconds": execution_duration,
        "image_paths": [],
        "cells": []
    }

    if os.path.exists(executed_nb_path):
        with open(executed_nb_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        image_counter = 1
        for i, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue

            cell_summary = {
                "cell_index": i,
                "outputs": [],
                "images": []
            }

            for output in cell.get("outputs", []):
                if output.output_type == "stream":
                    cell_summary["outputs"].append({
                        "type": output.name,
                        "text": output.text
                    })
                elif output.output_type in ("execute_result", "display_data"):
                    if "text/plain" in output.data:
                        cell_summary["outputs"].append({
                            "type": "text",
                            "text": output.data["text/plain"]
                        })

                    if "image/png" in output.data:
                        image_data = output.data["image/png"]
                        image_filename = f"cell_{i}_img_{image_counter}.png"
                        image_path = os.path.join(images_dir, image_filename)

                        with open(image_path, "wb") as img_file:
                            img_file.write(base64.b64decode(image_data))

                        rel_image_path = os.path.join("images", image_filename)
                        cell_summary["images"].append(rel_image_path)
                        summary["image_paths"].append(os.path.abspath(image_path))
                        image_counter += 1
                elif output.output_type == "error":
                    cell_summary["outputs"].append({
                        "type": "error",
                        "ename": output.ename,
                        "evalue": output.evalue,
                        "traceback": output.traceback
                    })

            summary["cells"].append(cell_summary)
    else:
        if success:
            success = False
            error_message = "Output notebook was not created."
            summary["success"] = False
            summary["error"] = error_message

    # Collect CPU/GPU usage metrics and attach before writing summary
    try:
        usage_metrics = _collect_usage_metrics()
    except Exception:
        usage_metrics = {"cpu_percent": None, "gpus": None}
    summary["usage_metrics"] = usage_metrics

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    result = {
        "success": summary.get("success", False),
        "output_dir": output_dir,
        "executed_notebook_path": summary.get("executed_notebook_path"),
        "summary_json_path": os.path.abspath(summary_path),
        "execution_time_seconds": summary.get("execution_time_seconds"),
        "error": summary.get("error"),
        "image_paths": summary.get("image_paths", [])
    }

    # Include usage metrics in the top-level result
    result["usage_metrics"] = summary.get("usage_metrics")

    return json.dumps(result, indent=2)