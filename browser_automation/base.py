import json
import getpass
import platform
import pathlib
import time

from selenium import webdriver


class BaseAutomation:
    def __init__(self, browser: str):
        browser = browser.strip().lower()

        if platform.system() == 'Windows':
            if browser == "chrome":
                user_data_dir = f"c:\\Users\\{getpass.getuser()}\\AppData\\Local\\Google\\Chrome\\User Data\\"
            elif browser == "edge":
                user_data_dir = f"c:\\Users\\{getpass.getuser()}\\AppData\\Local\\Microsoft\\Edge\\User Data\\"
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        if  browser == "chrome":
            from selenium.webdriver.chrome.options import Options
            self.browser_options = Options()
            # browser_options.add_argument("--disable-web-security")
            # browser_options.add_argument("--allow-running-insecure-content")
            self.browser_options.set_capability('goog:loggingPrefs', {'performance': 'ALL'})
            self.browser_options.add_argument('--no-sandbox')
            self.browser_options.add_argument("--disable-extensions")
            self.browser_options.add_argument("--disable-dev-shm-usage")
            self.browser_options.add_argument(f"--user-data-dir={user_data_dir}")
            self.driver_class = webdriver.Chrome
        elif browser == "edge":
            from selenium.webdriver.edge.options import Options
            self.browser_options = Options()
            # browser_options.add_argument("--disable-web-security")
            # browser_options.add_argument("--allow-running-insecure-content")
            self.browser_options.use_chromium = True
            self.browser_options.set_capability('goog:loggingPrefs', {'performance': 'ALL'})
            self.browser_options.add_argument('--no-sandbox')
            self.browser_options.add_argument("--disable-extensions")
            self.browser_options.add_argument("--disable-dev-shm-usage")
            self.browser_options.add_argument(f"--user-data-dir={user_data_dir}")
            self.browser_options.add_argument("profile-directory=Default")
            self.driver_class = webdriver.Edge
        else:
            raise NotImplementedError

        # print(self.driver_class)
        self.output_dir = ""
        self.service = "undefined"
        self.service_url = ""
        self.driver = None

    def automation(self, data, conversation_id, adv_prompt, multi_turn=True):
        if self.service == "undefined":
            raise NotImplemented()

        output_dir = pathlib.Path(__file__).parent.absolute() / "output" / self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{conversation_id}.json"
        if output_file.exists():
            return

        print(self.service, conversation_id)
        print(self.driver_class)
        self.driver = self.driver_class(options=self.browser_options)
        time.sleep(1)
        self.driver.get(self.service_url)
        time.sleep(1)
        print("!!!!", adv_prompt)
        try:
            conversations = data["conversations"]

            prompts = []
            if multi_turn:
                for conversation in conversations:
                    if conversation["role"] == "user":
                        prompts.append(conversation["content"])
                    if len(prompts) >= 2:
                        break
                prompts.append(adv_prompt)
            else:
                assert conversations
                assert conversations[0]["role"] == "user"
                prompts.append(conversations[0]["content"] + '\n\n' + adv_prompt)

            responses = []
            for i, prompt in enumerate(prompts):
                print(prompt)
                self.send_request(prompt)
                response = self.get_response(i)
                responses.append(response)
            data["context"] = data.pop("conversations") # change key name to align with downstream tasks
            data["result"] = [{
                "suffix": adv_prompt,
                "response": [{
                    "text": responses[-1]
                }],
            }]
            json.dump(data, output_file.open("w"), indent=4)

        except Exception as e:
            self.driver.close()
            time.sleep(1)
            raise e

        self.driver.close()
        time.sleep(1)


    def send_request(self, prompt):
        raise NotImplemented()


    def get_response(self, index):
        raise NotImplemented()