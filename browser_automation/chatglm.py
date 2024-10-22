import time

from selenium.webdriver import Keys, ActionChains
from selenium.webdriver.common.by import By

from base import BaseAutomation


class ChatGLM(BaseAutomation):
    def __init__(self, browser, output_dir="chatglm"):
        super(ChatGLM, self).__init__(browser)
        self.output_dir = output_dir
        self.service = "chatglm"
        self.service_url = "https://chatglm.cn/main/alltoolsdetail?lang=en"


    def send_request(self, prompt):
        chat_textarea = self.driver.find_element(By.TAG_NAME, "textarea")
        chat_textarea.clear()
        for part in prompt.split('\n'):
            chat_textarea.send_keys(part)
            ActionChains(self.driver).key_down(Keys.SHIFT).key_down(Keys.ENTER).key_up(Keys.SHIFT).key_up(
                Keys.ENTER).perform()

        button = self.driver.find_element(By.CSS_SELECTOR, "img.enter_icon")
        button.click()

    def get_response(self, index):
        data = None
        while True:
            time.sleep(2)
            elements = self.driver.find_elements(By.CSS_SELECTOR, "div.markdown-body")
            if index < len(elements):
                if data == elements[index].get_attribute('innerHTML'):
                    return data
                data = elements[index].get_attribute('innerHTML')
