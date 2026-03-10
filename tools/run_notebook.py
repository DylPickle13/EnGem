import os
import json
import base64
import time
import nbformat
import papermill as pm
from pathlib import Path
from typing import Any, Optional


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


def _normalize_source(source: Any) -> str:
    if isinstance(source, list):
        source_lines = [str(line) for line in source]
        if any(line.endswith("\n") for line in source_lines):
            return "".join(source_lines)
        return "\n".join(source_lines)
    if source is None:
        return ""
    return str(source)


def _sanitize_cell_metadata(metadata: Any) -> dict[str, Any]:
    if isinstance(metadata, dict):
        return dict(metadata)
    return {}


def _build_notebook_node(raw_notebook: dict[str, Any], parameters: dict[str, Any]) -> nbformat.NotebookNode:
    cells = []
    has_parameters_cell = False

    for raw_cell in raw_notebook.get("cells", []):
        if not isinstance(raw_cell, dict):
            continue

        cell_type = raw_cell.get("cell_type", "code")
        metadata = _sanitize_cell_metadata(raw_cell.get("metadata"))
        tags = metadata.get("tags")
        if isinstance(tags, list) and "parameters" in tags:
            has_parameters_cell = True

        source = _normalize_source(raw_cell.get("source", ""))
        if cell_type == "markdown":
            cell = nbformat.v4.new_markdown_cell(source=source, metadata=metadata)
        elif cell_type == "raw":
            cell = nbformat.v4.new_raw_cell(source=source, metadata=metadata)
        else:
            cell = nbformat.v4.new_code_cell(source=source, metadata=metadata)
            execution_count = raw_cell.get("execution_count")
            cell.execution_count = execution_count if isinstance(execution_count, int) else None
            raw_outputs = raw_cell.get("outputs")
            cell.outputs = raw_outputs if isinstance(raw_outputs, list) else []

        cells.append(cell)

    if not has_parameters_cell and parameters:
        parameter_lines = ["# Parameters"]
        for name, value in parameters.items():
            parameter_lines.append(f"{name} = {json.dumps(value)}")
        parameter_cell = nbformat.v4.new_code_cell(
            source="\n".join(parameter_lines),
            metadata={"tags": ["parameters"]},
        )
        cells.insert(0, parameter_cell)

    notebook_metadata = _sanitize_cell_metadata(raw_notebook.get("metadata"))
    notebook = nbformat.v4.new_notebook(cells=cells, metadata=notebook_metadata)

    return notebook


def _prepare_notebook_for_execution(
    nb_path: Path,
    output_dir: str,
    parameters: dict[str, Any],
) -> str:
    with open(nb_path, "r", encoding="utf-8") as f:
        raw_notebook = json.load(f)

    notebook = _build_notebook_node(raw_notebook, parameters)
    prepared_nb_path = os.path.join(output_dir, "prepared.ipynb")
    with open(prepared_nb_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    return prepared_nb_path


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
    # Disable execution timeout by passing None
    timeout = None

    start_time = time.time()
    success = True
    error_message = None

    try:
        prepared_nb_path = _prepare_notebook_for_execution(nb_path, output_dir, params)
        # Run from the notebook folder so relative writes go to expected paths.
        notebook_dir = str(nb_path.parent)
        old_cwd = os.getcwd()
        try:
            os.chdir(notebook_dir)
            pm.execute_notebook(
                prepared_nb_path,
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

    
    return json.dumps(result, indent=2)