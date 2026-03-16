# run_python tool instructions

Use this tool when you need to execute Python code directly in the repository environment.

What this tool does:
- Executes the provided Python code with the repo root as working directory.
- Returns output in this wrapper format:
  - Code run:
  - Output:
- Captures stdout and stderr.
- Moves newly created root-level files with known media/document extensions into generated_files/.
- Appends a Generated artifacts section with absolute paths when artifacts are moved.

How to call it well:
- Send complete runnable Python code in one call.
- Use print statements for key intermediate and final results.
- Keep scripts focused and deterministic.
- When creating downloadable outputs, write to generated_files/ and print the final absolute path.

Important constraints:
- Syntax errors are returned directly.
- Very long stdout is truncated.
- If you rely on files from earlier steps, verify they exist before using them.

High quality prompting pattern:
- State exact objective.
- Include all inputs and assumptions in code.
- Print a concise final summary plus any output file paths.