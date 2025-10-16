from playwright.sync_api import sync_playwright
import json

USERNAME = "-"
PASSWORD = "-"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)  
    context = browser.new_context()
    page = context.new_page()
    
    page.goto("https://www.linkedin.com/login")
    
    page.fill('input[name="session_key"]', USERNAME)
    page.fill('input[name="session_password"]', PASSWORD)
    page.click('button[type="submit"]')
    
    page.wait_for_timeout(5000)  
    
    
    cookies = context.cookies()
    with open("linkedin_cookies.json", "w") as f:
        json.dump(cookies, f)
    
    print("ok")
    browser.close()


