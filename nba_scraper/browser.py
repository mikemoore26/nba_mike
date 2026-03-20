# nba_scraper/browser.py
from __future__ import annotations

import random
import time
from typing import Optional

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

def create_driver(*, headless: bool = True) -> webdriver.Chrome:
    """
    Create a Chrome driver configured for scraping.
    pageLoadStrategy='eager' prevents long hangs waiting for every asset.
    Blocks images/fonts for speed.
    """
    options = Options()

    if headless:
        options.add_argument("--headless=new")
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


def fetch_html(
    driver: Optional[webdriver.Chrome],
    url: str,
    *,
    retries: int = 3,
    retry_delay: float = 5.0,
    wait_for_css_any: Optional[list[str]] = None,
    timeout: float = 15.0,
    hard_restart_delay: float = 2.0,
) -> tuple[str, webdriver.Chrome]:
    """
    Loads a URL and returns (html, driver). If driver is None, creates one.
    Centralize ALL retry/backoff/driver-restart logic here.
    """
    if driver is None:
        driver = create_driver()

    wait_for_css_any = wait_for_css_any or ["body"]

    for attempt in range(1, retries + 1):
        try:
            driver.get(url)

            wait_for_any(driver, wait_for_css_any, timeout=timeout)

            # stop extra asset loads (eager already helps)
            try:
                driver.execute_script("window.stop();")
            except Exception:
                pass

            html = driver.page_source or ""
            return html, driver

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

            time.sleep(hard_restart_delay)
            driver = create_driver()

            # jittered backoff
            base = (hard_restart_delay if hard_restart else retry_delay)
            sleep_s = base * attempt + random.uniform(0.25, 0.9)
            time.sleep(sleep_s)

    return "", driver

def get_page_source(url: str, retries: int = 3, delay: float = 5.0) -> str:
    """
    Backwards-compatible wrapper: fetches HTML with a fresh driver per call.
    """
    driver = None
    try:
        html, driver = fetch_html(
            driver,
            url,
            retries=retries,
            retry_delay=delay,
            wait_for_css_any=["body"],
        )
        return html
    finally:
        try:
            if driver:
                driver.quit()
        except Exception:
            pass
