from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class Tool:
    def __init__(self, description, name, parameters):
        self.type = "function"
        self.description = description  # A description of what the function does, used by the model to choose when and how to call the function.
        self.name = name  # The name of the function to be called. Must be a-z, A-Z, 0-9, or contain underscores and dashes, with a maximum length of 64.
        self.parameters = parameters  # The parameters the functions accepts, described as a JSON Schema object. See the guide for examples, and the JSON Schema reference for documentation about the format. Omitting parameters defines a function with an empty parameter list.

    @abstractmethod
    def func(self):
        pass

    @abstractmethod
    def to_dict(self):
        pass

    @abstractmethod
    def load_from_dict(cls, dict_data):
        pass

class WebCrawlTool(Tool):
    """Open a single web page for crawling"""

    def __init__(self):
        description = "Open a single web page for crawling"
        name = "web_crawl"
        parameters = {
            "url": {
                "type": "string",
                "description": "The URL of the web page to be crawled",
            }
        }
        super(WebCrawlTool, self).__init__(description, name, parameters)

    def func(self, url):
        print(f"crawling {url} ......")
        content = ""
        """Crawling content from url may need to be carried out according to different websites, such as wiki, baidu, zhihu, etc."""
        driver = webdriver.Chrome()
        try:
            """open url"""
            driver.get(url)

            """wait 20 second"""
            wait = WebDriverWait(driver, 20)
            wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))

            """crawl code"""
            page_source = driver.page_source

            """parse"""
            soup = BeautifulSoup(page_source, "html.parser")

            """concatenate"""
            for paragraph in soup.find_all("p"):
                content = f"{content}\n{paragraph.get_text()}"
        except Exception as e:
            print("Error:", e)
        finally:
            """quit"""
            driver.quit()
        return {"content": content.strip()}