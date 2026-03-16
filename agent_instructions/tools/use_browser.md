# use_browser tool instructions

Use this tool for interactive browser actions that require page rendering, clicks, form input, or navigation.

What this tool does:
- Launches an automated browser session.
- Runs the provided prompt through the browser-use loop.
- Returns a textual result of actions/findings.
- Closes the browser at the end of the call.

How to call it well:
- Provide a specific URL or starting page.
- Give explicit step-by-step actions in order.
- Define clear extraction targets (exact fields, values, or links).
- Ask for a concise final report of what was found.

Important constraints:
- Browser state does not persist across calls.
- Prefer one focused browsing task per call.
- If multiple independent browser operations are needed, split them into separate calls.

High quality prompting pattern:
- Start location.
- Action sequence.
- Success criteria.
- Exact output format to return.