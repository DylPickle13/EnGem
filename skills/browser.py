# from playwright.sync_api import sync_playwright


# def start_browser(url: str = "https://www.google.com"):
#     """Start a Playwright browser and return (playwright, browser, page, actions).

#     The returned objects allow the caller to interact with the browser and
#     later call `close_browser(playwright, browser)` to shut it down.
#     """
#     p = sync_playwright().start()
#     browser = p.chromium.launch(headless=False)
#     page = browser.new_page()
#     page.goto(url, wait_until="domcontentloaded")

#     actions = page.evaluate(r"""
#         () => {
#             const clean = (value) => (value || '').toString().trim().replace(/\s+/g, ' ');
#             const out = [];

#             const links = Array.from(document.querySelectorAll('a[href]')).slice(0, 25);
#             for (const link of links) {
#                 const label = clean(link.innerText) || clean(link.getAttribute('aria-label')) || clean(link.getAttribute('title')) || clean(link.href);
#                 if (label) out.push(`click link: ${label}`);
#             }

#             const buttons = Array.from(document.querySelectorAll('button, [role="button"], input[type="button"], input[type="submit"]')).slice(0, 25);
#             for (const button of buttons) {
#                 const label = clean(button.innerText) || clean(button.value) || clean(button.getAttribute('aria-label')) || clean(button.getAttribute('name'));
#                 if (label) out.push(`click button: ${label}`);
#             }

#             const fields = Array.from(document.querySelectorAll('input, textarea, select')).slice(0, 25);
#             for (const field of fields) {
#                 const label = clean(field.getAttribute('aria-label')) || clean(field.getAttribute('name')) || clean(field.getAttribute('placeholder')) || clean(field.id);
#                 if (label) out.push(`fill field: ${label}`);
#             }

#             return Array.from(new Set(out));
#         }
#         """)

#     while True:
#         i = 1


# if __name__ == "__main__":
#     p, browser, page, actions = start_browser("https://www.google.com")
#     print("Available actions:")
#     for action in actions:
#         print(f"- {action}")