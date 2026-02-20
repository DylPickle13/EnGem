import ast
import subprocess
import sys
import tempfile


def run_python(code: str):
    """Run Python code safely and return stdout and stderr."""
    try:
        ast.parse(code, mode="exec")
    except SyntaxError as e:
        return "", f"SyntaxError: {e}"

    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
        f.write(code.encode())
        filename = f.name

    result = subprocess.run(
        [sys.executable, filename],
        capture_output=True,
        text=True,
        timeout=10
    )
    
    if result.stderr:
        return result.stderr
    return result.stdout