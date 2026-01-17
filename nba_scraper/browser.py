# nba_scraper/browser.py
from __future__ import annotations

import time
from typing import Iterable, Tuple, Optional

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException


from selenium import webdriver
from selenium.webdriver.chrome.options import Options


def log(msg: str, level: str = "INFO"):
    print(f"[{level}] {msg}")

def wait_for_any(driver, css_selectors: list[str], timeout: float = 15.0) -> bool:
    """
    Poll until ANY selector exists in the DOM. Returns True if found, else False.
    """
    end = time.time() + timeout
    while time.time() < end:
        for css in css_selectors:
            try:
                if driver.find_elements("css selector", css):
                    return True
            except Exception:
                pass
        time.sleep(0.25)
    return False

def create_driver() -> webdriver.Chrome:
    """
    Create a headless Chrome driver configured for scraping.
    pageLoadStrategy='eager' prevents long hangs waiting for every asset.
    Blocks images/fonts for speed + fewer renderer hangs.
    """
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")

    # Return when DOM is ready-ish, not after every asset
    options.set_capability("pageLoadStrategy", "eager")

    # Block heavy resources
    prefs = {
        "profile.managed_default_content_settings.images": 2,
        "profile.managed_default_content_settings.fonts": 2,
        "profile.default_content_setting_values.notifications": 2,
    }
    options.add_experimental_option("prefs", prefs)

    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(30)
    driver.set_script_timeout(30)
    return driver

def fetch_html(driver, url: str, retries: int = 3, retry_delay: float = 5.0):
    for attempt in range(1, retries + 1):
        try:
            driver.set_page_load_timeout(25)
            driver.get(url)

            wait_for_any(driver, ["#player_game_log_reg", "body"], timeout=15)

            try:
                driver.execute_script("window.stop();")
            except Exception:
                pass

            return driver.page_source or "", driver

        except Exception as e:
            msg = str(e)
            log(f"Error loading {url} (attempt {attempt}/{retries}): {e}", "WARN")

            hard_restart = (
                "HTTPConnectionPool(host='localhost'" in msg
                or "Read timed out" in msg
                or "MaxRetryError" in msg
                or "chrome not reachable" in msg
                or "disconnected" in msg
                or "Timed out receiving message from renderer" in msg
            )

            try:
                driver.execute_script("window.stop();")
            except Exception:
                pass

            try:
                driver.quit()
            except Exception:
                pass

            time.sleep(2)
            driver = create_driver()

            time.sleep(2 if hard_restart else retry_delay)

    return "", driver

def get_page_source(url: str, retries: int = 3, delay: int = 5) -> str:
    """
    Backwards-compatible wrapper: fetches HTML with a fresh driver per call.
    (Your older code can keep using this, but the optimized path is fetch_html()).
    """
    driver = None
    try:
        html, driver = fetch_html(
            driver=None,
            url=url,
            retries=retries,
            delay=delay,
            wait_seconds=10,
            wait_for_css_any=["body"],
        )
        return html
    finally:
        try:
            if driver:
                driver.quit()
        except Exception:
            pass
