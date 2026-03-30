# run_notebook tool instructions

Use this tool to execute a Jupyter notebook and capture execution outputs in a structured way.

What this tool does:
- Resolves notebook path (absolute, repo-relative, filename search fallback).
- Executes notebook via papermill.
- Creates a timestamped results folder.
- Writes executed notebook and summary.json.
- Extracts PNG outputs from notebook cells.
- Returns JSON with success, output_dir, executed_notebook_path, summary_json_path, execution_time_seconds, error, image_paths.

How to call it well:
- Pass a precise notebook path.
- If notebook depends on local files, ensure paths are valid from notebook context.
- After execution, inspect success and error first.
- Use summary_json_path and image_paths for downstream steps.

Important constraints:
- Tool returns JSON text, not rendered notebook output.
- Missing notebook path returns success=false with an error message.
- Execution may take time depending on notebook workload.
- Notebook execution is capped at 1 hour of wall-clock time; if it exceeds that limit, the tool returns success=false with a timeout error.

High quality prompting pattern:
- Specify notebook path.
- State what result artifacts are needed.
- Request extraction of key metrics from summary.json when relevant.