import time
import threading
from pathlib import Path
from google import genai
from google.genai import types
from google.genai.types import Content, Part
from playwright.sync_api import Browser, Page, Playwright, sync_playwright

from config import get_paid_gemini_api_key as get_paid_gemini_api_key, MEDIUM_MODEL as MEDIUM_MODEL
from api_backoff import call_with_exponential_backoff

BROWSER_FILE = Path(__file__).parent / "agent_instructions/browser.md"
SCREEN_WIDTH = 1440
SCREEN_HEIGHT = 900
TURN_LIMIT = 50
MODEL_NAME = MEDIUM_MODEL
ALT_MODEL_NAME = ""
DEBUG_SCROLL = False
DEFAULT_SCROLL_DELTA = 600
DEFAULT_DOCUMENT_SCROLL_AMOUNT = 800

_PLAYWRIGHT_INSTANCES: dict[int, Playwright] = {}
_BROWSER_INSTANCES: dict[int, Browser] = {}


def _select_model_for_loop(prompt: str) -> str:
    """Pick a single model for the whole loop to preserve tool-call turn consistency."""
    model_pool = [MODEL_NAME]
    if ALT_MODEL_NAME:
        model_pool.append(ALT_MODEL_NAME)
    if len(model_pool) == 1:
        return model_pool[0]
    key = f"{threading.get_ident()}::{prompt}"
    return model_pool[abs(hash(key)) % len(model_pool)]


def denormalize_x(x: int, screen_width: int) -> int:
    """Convert normalized x coordinate (0-1000) to actual pixel coordinate."""
    return int(max(0, min(x, 1000)) / 1000 * screen_width)

def denormalize_y(y: int, screen_height: int) -> int:
    """Convert normalized y coordinate (0-1000) to actual pixel coordinate."""
    return int(max(0, min(y, 1000)) / 1000 * screen_height)


def _actual_coordinates(args: dict, screen_width: int, screen_height: int) -> tuple[int, int]:
    x = int(args.get("x", 0))
    y = int(args.get("y", 0))
    return denormalize_x(x, screen_width), denormalize_y(y, screen_height)


def _extract_scroll_deltas(args: dict) -> tuple[int, int]:
    delta_x = args.get("delta_x")
    delta_y = args.get("delta_y")

    if delta_x is None:
        delta_x = args.get("scroll_x", args.get("wheel_x", 0))
    if delta_y is None:
        delta_y = args.get("scroll_y", args.get("wheel_y"))

    amount = args.get("amount", args.get("scroll_amount", args.get("pixels")))
    if delta_y is None and amount is not None:
        delta_y = amount
    if delta_y is None:
        delta_y = 0

    direction = str(args.get("direction", "")).lower().strip()
    if direction in {"up", "down", "left", "right"}:
        if delta_y:
            magnitude = abs(int(float(delta_y)))
        else:
            raw_magnitude = args.get("magnitude", args.get("amount", DEFAULT_SCROLL_DELTA))
            magnitude = abs(int(float(raw_magnitude)))
        if direction == "up":
            delta_x, delta_y = 0, -magnitude
        elif direction == "down":
            delta_x, delta_y = 0, magnitude
        elif direction == "left":
            delta_x, delta_y = -magnitude, 0
        elif direction == "right":
            delta_x, delta_y = magnitude, 0

    return int(float(delta_x or 0)), int(float(delta_y or 0))


def _is_navigation_context_error(error: Exception) -> bool:
    message = str(error).lower()
    return "execution context was destroyed" in message or "most likely because of a navigation" in message


def _evaluate_with_navigation_retry(page: Page, script: str, payload: dict):
    try:
        return page.evaluate(script, payload)
    except Exception as error:
        if not _is_navigation_context_error(error):
            raise

        try:
            page.wait_for_load_state("domcontentloaded", timeout=5000)
            page.wait_for_timeout(150)
        except Exception:
            pass

        return page.evaluate(script, payload)


def _capture_scroll_state(page: Page, x: int | None = None, y: int | None = None) -> dict:
    try:
        return _evaluate_with_navigation_retry(
            page,
            """({x, y}) => {
            const isScrollable = (el) => {
                if (!el) return false;
                return (
                    (el.scrollHeight > el.clientHeight || el.scrollWidth > el.clientWidth) &&
                    (getComputedStyle(el).overflowY !== 'hidden' || getComputedStyle(el).overflowX !== 'hidden')
                );
            };

            const findScrollable = (start) => {
                let node = start;
                while (node && node !== document.body && node !== document.documentElement) {
                    if (isScrollable(node)) return node;
                    node = node.parentElement;
                }
                return document.scrollingElement || document.documentElement || document.body;
            };

            const docEl = document.scrollingElement || document.documentElement || document.body;
            const state = {
                docLeft: docEl ? docEl.scrollLeft : 0,
                docTop: docEl ? docEl.scrollTop : 0,
                targetLeft: null,
                targetTop: null,
            };

            if (typeof x === 'number' && typeof y === 'number') {
                const target = document.elementFromPoint(x, y);
                const scroller = findScrollable(target);
                if (scroller) {
                    state.targetLeft = scroller.scrollLeft;
                    state.targetTop = scroller.scrollTop;
                }
            }

            return state;
        }""",
            {"x": x, "y": y},
        )
    except Exception as error:
        if DEBUG_SCROLL:
            print(f"[scroll-debug] failed to capture scroll state: {error}")
        return {
            "docLeft": None,
            "docTop": None,
            "targetLeft": None,
            "targetTop": None,
            "error": str(error),
        }


def _programmatic_scroll(page: Page, delta_x: int, delta_y: int, x: int | None = None, y: int | None = None) -> None:
    _evaluate_with_navigation_retry(
        page,
        """({dx, dy, x, y}) => {
            const canScroll = (el) => {
                if (!el) return false;
                return el.scrollHeight > el.clientHeight || el.scrollWidth > el.clientWidth;
            };

            const findScrollable = (start) => {
                let node = start;
                while (node && node !== document.body && node !== document.documentElement) {
                    if (canScroll(node)) return node;
                    node = node.parentElement;
                }
                return document.scrollingElement || document.documentElement || document.body;
            };

            let scroller = document.scrollingElement || document.documentElement || document.body;
            if (typeof x === 'number' && typeof y === 'number') {
                const target = document.elementFromPoint(x, y);
                scroller = findScrollable(target);
            }

            const beforeLeft = scroller.scrollLeft;
            const beforeTop = scroller.scrollTop;
            if (typeof scroller.scrollBy === 'function') {
                scroller.scrollBy(dx, dy);
            } else {
                scroller.scrollLeft += dx;
                scroller.scrollTop += dy;
            }

            const didMove = beforeLeft !== scroller.scrollLeft || beforeTop !== scroller.scrollTop;
            if (!didMove) {
                window.scrollBy(dx, dy);
            }
        }""",
        {"dx": delta_x, "dy": delta_y, "x": x, "y": y},
    )


def _resolve_scroll_target(args: dict, screen_width: int, screen_height: int) -> tuple[int, int]:
    if "x" in args and "y" in args:
        return _actual_coordinates(args, screen_width, screen_height)
    return screen_width // 2, screen_height // 2


def _is_scroll_state_changed(before_state: dict, after_state: dict) -> bool:
    numeric_before = isinstance(before_state.get("docTop"), (int, float))
    numeric_after = isinstance(after_state.get("docTop"), (int, float))
    if not (numeric_before and numeric_after):
        return False

    return (
        before_state.get("docTop") != after_state.get("docTop")
        or before_state.get("docLeft") != after_state.get("docLeft")
        or before_state.get("targetTop") != after_state.get("targetTop")
        or before_state.get("targetLeft") != after_state.get("targetLeft")
    )


def _execute_single_action(
    page: Page,
    fname: str,
    args: dict,
    screen_width: int,
    screen_height: int,
) -> dict:
    if fname == "open_web_browser":
        target_url = args.get("url") or args.get("target_url") or args.get("website")
        if target_url:
            page.goto(target_url)
        return {}

    if fname in {"navigate", "navigate_to", "open_url", "go_to"}:
        target_url = args.get("url")
        if not target_url:
            return {"error": "Missing url argument"}
        page.goto(target_url)
        return {}

    if fname == "extract_elements":
        selector = args.get("selector")
        if not selector:
            return {"error": "Missing selector argument"}

        raw_attributes = args.get("attributes", [])
        if isinstance(raw_attributes, str):
            attributes = [raw_attributes]
        elif isinstance(raw_attributes, (list, tuple)):
            attributes = [str(attribute) for attribute in raw_attributes if attribute]
        else:
            attributes = []

        include_text = bool(args.get("include_text", not attributes))
        limit = int(args.get("limit", args.get("max_elements", 50)))
        if limit < 1:
            limit = 1

        extracted = _evaluate_with_navigation_retry(
            page,
            """({selector, attributes, includeText, limit}) => {
            const nodes = Array.from(document.querySelectorAll(selector)).slice(0, limit);
            return nodes.map((node) => {
                const item = {};
                if (includeText) {
                    item.text = (node.textContent || '').trim();
                }

                for (const attribute of attributes) {
                    item[attribute] = node.getAttribute(attribute);
                }

                if (!includeText && attributes.length === 0) {
                    item.text = (node.textContent || '').trim();
                }

                return item;
            });
        }""",
            {
                "selector": selector,
                "attributes": attributes,
                "includeText": include_text,
                "limit": limit,
            },
        )

        return {
            "selector": selector,
            "count": len(extracted),
            "elements": extracted,
        }

    if fname == "click_at":
        actual_x, actual_y = _actual_coordinates(args, screen_width, screen_height)
        button = args.get("button", "left")
        click_count = int(args.get("click_count", 1))
        page.mouse.click(actual_x, actual_y, button=button, click_count=click_count)
        return {}

    if fname == "double_click_at":
        actual_x, actual_y = _actual_coordinates(args, screen_width, screen_height)
        page.mouse.dblclick(actual_x, actual_y)
        return {}

    if fname == "hover_at":
        actual_x, actual_y = _actual_coordinates(args, screen_width, screen_height)
        page.mouse.move(actual_x, actual_y)
        return {}

    if fname == "type_text_at":
        actual_x, actual_y = _actual_coordinates(args, screen_width, screen_height)
        text = args.get("text", "")
        press_enter = args.get("press_enter", False)
        should_clear = args.get("clear_first", True)

        page.mouse.click(actual_x, actual_y)
        if should_clear:
            page.keyboard.press("Meta+A")
            page.keyboard.press("Backspace")
        page.keyboard.type(text)
        if press_enter:
            page.keyboard.press("Enter")
        return {}

    if fname in {"press_key", "keyboard_press"}:
        key = args.get("key")
        if not key:
            return {"error": "Missing key argument"}
        page.keyboard.press(key)
        return {}

    if fname in {
        "scroll_by",
        "scroll",
        "scroll_at",
        "scroll_down",
        "scroll_up",
        "scroll_document",
        "scroll_page",
    }:
        target_x, target_y = _resolve_scroll_target(args, screen_width, screen_height)
        page.mouse.move(target_x, target_y)

        # Give keyboard fallbacks a focused context.
        page.mouse.click(target_x, target_y)

        # Support multiple payload shapes from different model/tool versions.
        delta_x, delta_y = _extract_scroll_deltas(args)

        if fname == "scroll_down" and delta_y == 0:
            delta_y = DEFAULT_SCROLL_DELTA
        elif fname == "scroll_up" and delta_y == 0:
            delta_y = -DEFAULT_SCROLL_DELTA

        # Document-oriented tools often use direction/amount without explicit deltas.
        if fname in {"scroll_document", "scroll_page"} and delta_x == 0 and delta_y == 0:
            direction = str(args.get("direction", "down")).lower().strip()
            amount = int(float(args.get("amount", args.get("scroll_amount", DEFAULT_DOCUMENT_SCROLL_AMOUNT))))
            delta_y = -abs(amount) if direction == "up" else abs(amount)

        before_state = _capture_scroll_state(page, target_x, target_y)
        page.mouse.wheel(delta_x, delta_y)
        page.wait_for_timeout(250)
        after_state = _capture_scroll_state(page, target_x, target_y)

        # Fallback for no-op wheel payloads.
        wheel_moved = _is_scroll_state_changed(before_state, after_state)
        if not wheel_moved:
            if delta_x == 0 and delta_y == 0:
                direction = str(args.get("direction", "down")).lower().strip()
                key = "PageUp" if direction == "up" or fname == "scroll_up" else "PageDown"
                page.keyboard.press(key)
            else:
                try:
                    _programmatic_scroll(page, delta_x, delta_y, target_x, target_y)
                except Exception as error:
                    if DEBUG_SCROLL:
                        print(f"[scroll-debug] programmatic scroll failed: {error}")


        return {}

    if fname == "wait":
        seconds = float(args.get("seconds", 1))
        time.sleep(max(0, seconds))
        return {}

    if fname == "wait_5_seconds":
        time.sleep(5)
        return {}

    if fname == "go_back":
        page.go_back()
        return {}

    if fname == "go_forward":
        page.go_forward()
        return {}

    if fname in {"reload", "refresh_page"}:
        page.reload()
        return {}

    print(f"Warning: Unimplemented or custom function {fname} with args={dict(args)}")
    return {"warning": f"Unimplemented or custom function: {fname}"}


def execute_function_calls(candidate, page: Page, screen_width: int, screen_height: int):
    results = []
    function_calls = []
    for part in candidate.content.parts:
        if part.function_call:
            function_calls.append(part.function_call)

    for function_call in function_calls:
        extra_fr_fields = {}
        # Check for safety decision
        if 'safety_decision' in function_call.args:
            safety_decision = function_call.args['safety_decision']
            safety_label = str(safety_decision.get("decision", "")).lower()

            if safety_label == "require_confirmation":
                extra_fr_fields["safety_acknowledgement"] = True

        fname = function_call.name
        args = function_call.args
        #print(f"  -> Executing: {fname}")

        try:
            action_result = _execute_single_action(page, fname, args, screen_width, screen_height)

            # Wait for potential navigations/renders
            page.wait_for_load_state(timeout=5000)

        except Exception as e:
            print(f"Error executing {fname}: {e}")
            action_result = {"error": str(e)}

        results.append((fname, action_result, extra_fr_fields))

    return results


def get_function_responses(page, results):
    screenshot_bytes = page.screenshot(type="png")
    current_url = page.url
    function_responses = []
    for name, result, extra_fr_fields in results:
        response_data = {"url": current_url}
        response_data.update(result)
        response_data.update(extra_fr_fields)
        function_responses.append(
            types.FunctionResponse(
                name=name,
                response=response_data,
                parts=[types.FunctionResponsePart(
                        inline_data=types.FunctionResponseBlob(
                            mime_type="image/png",
                            data=screenshot_bytes))
                ]
            )
        )
    return function_responses


def create_client() -> genai.Client:
    return genai.Client(api_key=get_paid_gemini_api_key())


def create_model_config() -> types.GenerateContentConfig:
    config = types.GenerateContentConfig(
        tools=[types.Tool(computer_use=types.ComputerUse(
            environment=types.Environment.ENVIRONMENT_BROWSER
        ))],
        thinking_config=types.ThinkingConfig(include_thoughts=True),
        system_instruction=BROWSER_FILE.read_text(encoding="utf-8")
    )
    return config


def setup_browser() -> tuple[Playwright, Browser, Page]:
    """
    Always start a fresh Playwright and Browser for the current thread.
    This avoids reusing existing browser instances while keeping the pattern
    that each browser is created on the calling thread.
    """
    tid = threading.get_ident()

    # If there are any leftover instances for this thread, close them first
    # to ensure we don't reuse or leak resources.
    old_br = _BROWSER_INSTANCES.pop(tid, None)
    old_pw = _PLAYWRIGHT_INSTANCES.pop(tid, None)
    if old_br:
        try:
            old_br.close()
        except Exception:
            pass
    if old_pw:
        try:
            old_pw.stop()
        except Exception:
            pass

    pw = sync_playwright().start()
    br = pw.chromium.launch(headless=False)

    # Do not store the new instances for reuse — callers should close them
    # when finished. Returning fresh instances ensures no reuse occurs.
    context = br.new_context(viewport={"width": SCREEN_WIDTH, "height": SCREEN_HEIGHT})
    page = context.new_page()
    return pw, br, page


def run_agent_loop(client: genai.Client, page: Page, prompt: str) -> str:
    #print("Prompt:", prompt)
    config = create_model_config()
    initial_screenshot = page.screenshot(type="png")
    model_to_use = _select_model_for_loop(prompt)
    #print(f"Using model for this loop: {model_to_use}")

    contents = [
        Content(role="user", parts=[
            Part(text=prompt),
            Part.from_bytes(data=initial_screenshot, mime_type='image/png')
        ])
    ]

    responses = ""
    for i in range(TURN_LIMIT):
        #print(f"\n--- Turn {i+1} ---")
        response = call_with_exponential_backoff(
            lambda: client.models.generate_content(
                model=model_to_use,
                contents=contents,
                config=config,
            ),
            description=f"Gemini browser agent turn {i + 1}",
        )

        candidate = response.candidates[0]
        contents.append(candidate.content)

        has_function_calls = any(part.function_call for part in candidate.content.parts)
        if not has_function_calls:
            #print("No function calls detected, ending agent loop.")
            responses += "".join(part.text for part in candidate.content.parts if part.text)
            break

        results = execute_function_calls(candidate, page, SCREEN_WIDTH, SCREEN_HEIGHT)

        function_responses = get_function_responses(page, results)

        contents.append(
            Content(role="user", parts=[Part(function_response=fr) for fr in function_responses])
        )
        responses += "".join(part.text for part in candidate.content.parts if part.text)
    return responses