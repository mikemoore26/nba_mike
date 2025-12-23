def get_page_source(url: str, retries: int = 3, delay: int = 5) -> str:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    import time
    
    """
    Load a page with headless Chrome, with retries on Selenium/driver errors.
    Returns HTML string, or "" if all attempts fail.
    """
    for attempt in range(1, retries + 1):
        options = Options()
        options.add_argument("--headless=new")   # run Chrome in headless mode
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")

        driver = webdriver.Chrome(options=options)
        try:
            driver.set_page_load_timeout(60)  # seconds
            print(f"[TRY {attempt}/{retries}] Accessing {url}")
            driver.get(url)
            html = driver.page_source
            return html
        except Exception as e:
            print(f"Error accessing {url} on attempt {attempt}: {e}")
            if attempt == retries:
                print(f"[FAIL] Giving up on {url}")
                return ""
            time.sleep(delay)
        finally:
            driver.quit()
