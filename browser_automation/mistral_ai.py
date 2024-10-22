import json
import time
from xxsubtype import bench

from selenium.webdriver import Keys, ActionChains
from selenium.webdriver.common.by import By

from base import BaseAutomation


class MistralAI(BaseAutomation):
    def __init__(self, browser, output_dir="mistral_ai"):
        super(MistralAI, self).__init__(browser)
        self.output_dir = output_dir
        self.service = "mistral_ai"
        self.service_url = "https://chat.mistral.ai/chat"


    def send_request(self, prompt):
        prompt = prompt.replace("\t", "    ")

        for part in prompt.split('\n'):
            while True:
                try:
                    chat_textarea = self.driver.find_element(By.TAG_NAME, "textarea")
                    break
                except:
                    time.sleep(1)
            if part:
                chat_textarea.send_keys(part)
            ActionChains(self.driver).key_down(Keys.SHIFT).key_down(Keys.ENTER).key_up(Keys.SHIFT).key_up(
                Keys.ENTER).perform()

        while True:
            try:
                button = self.driver.find_element(By.CSS_SELECTOR, "button[type=submit]")
                break
            except:
                time.sleep(1)
        button.click()


    def get_response(self, index):
        data = None
        while True:
            time.sleep(2)
            elements = self.driver.find_elements(By.CSS_SELECTOR, "div.prose")
            if index < len(elements):
                if data == elements[index].get_attribute('innerHTML'):
                    try:
                        browser_log = self.driver.get_log('performance')
                        # print(browser_log)
                        events = [json.loads(entry['message'])['message'] for entry in browser_log]
                        chat_event = None
                        for event in reversed(events):
                            # print(event)
                            if 'Network.response' in event['method'] and event.get('params', {}).get('response', {}).get('url', '') == 'https://chat.mistral.ai/api/chat':
                                chat_event = event
                                break
                        if chat_event:
                            # print(chat_event)
                            body = self.driver.execute_cdp_cmd('Network.getResponseBody', {'requestId': chat_event['params']["requestId"]})
                            lines = body['body'].split('\n')
                            parsed_body = ""
                            for line in lines:
                                if line.startswith('0:'):
                                    parsed_body += line[3:-1]

                            return parsed_body
                    except Exception as e:
                        print("error:", e)
                        return data

                data = elements[index].get_attribute('innerHTML')
