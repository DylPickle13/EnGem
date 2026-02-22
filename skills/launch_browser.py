from playwright.sync_api import sync_playwright

def launch_browser():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto("https://www.google.com")
        print(f"Page title: {page.title()}")
        print("Browser launched. It will remain open until you manually close it or the script is terminated.")
        # You can add further Playwright actions here.
        # To keep the browser open for interaction, we'll wait for user input.
        input("Press Enter to close the browser...")
        browser.close()

if __name__ == "__main__":
    launch_browser()