import requests
import math
import os
from typing import List
from bs4 import BeautifulSoup
import re
import cloudscraper
from dotenv import load_dotenv
from openai import OpenAI
import time
import sys
import json
from jinja2 import Environment, FileSystemLoader


class WebScraper:
    def __init__(self):
        load_dotenv()
        self.google_search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
        self.google_search_engine_key = os.getenv("GOOGLE_SEARCH_ENGINE_KEY")

    def retrieve_serp_urls(self, question: str, num_search_results: int) -> List[str]:
        if num_search_results > 100:
            raise NotImplementedError('Google Custom Search API supports a max of 100 results')
        elif num_search_results > 10:
            num = 10
            calls_to_make = math.ceil(num_search_results / 10)
        else:
            calls_to_make = 1
            num = num_search_results

        start_item = 1
        items_to_return = []

        while calls_to_make > 0:
            items = self.get_urls(question, start_item, num)
            items_to_return.extend(items)
            calls_to_make -= 1
            start_item += num
            leftover = num_search_results - start_item + 1
            if 0 < leftover < 10:
                num = leftover

        return items_to_return

    def get_urls(self, question: str, start_item: int, num: int) -> List[str]:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "cx": self.google_search_engine_id,
            "q": f"{question} -filetype:pdf -filetype:xml",
            "hl": "nl",
            "gl": "be",
            "num": num,
            "cr": "Belgium",
            "lr": "lang_be|lang_en",
            "key": self.google_search_engine_key,
            "start": start_item
        }

        response = requests.get(url, params=params)
        results = response.json()['items']

        #print(f"{len(results)} are retrieved")
        return [item['link'] for item in results]

    def crawl_urls(self, urls: List[str], temp_file_path: str):
        docs = []
        for url in urls:
            static_content = self.get_static_content(url, 10)

            if static_content and len(static_content) > 100:
                text = self.extract_text_from_html(static_content)
            else:
                dynamic_content = self.get_dynamic_content(url, 10)
                if dynamic_content is None:
                    print("None returned")
                elif dynamic_content:
                    text = self.extract_text_from_html(dynamic_content)
                else:
                    print("Failed to retrieve content from the website.")
                    continue

            #print(url)
            docs.append({"url": url, "content": text})

        with open(temp_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(docs, json_file, ensure_ascii=False, indent=4)

    def get_static_content(self, url, timeout):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response.content
        except requests.RequestException as e:
            print(f"Request to {url} failed: {e}")
            return None

    def get_dynamic_content(self, url, timeout):
        try:
            scraper = cloudscraper.create_scraper(browser={'browser': 'firefox', 'platform': 'windows', 'mobile': False})
            response = scraper.get(url, timeout=timeout)
            return response.content
        except Exception as e:
            print(f"Request to {url} failed: {e}")
            return None

    def extract_text_from_html(self, html_content):
        soup = BeautifulSoup(html_content, 'html.parser')

        content_tags = ['p', 'article', 'section']
        excluded_classes = ['header', 'footer', 'nav', 'sidebar', 'menu', 'breadcrumb', 'pagination', 'legal', 'advertisement']
        excluded_ids = ['header', 'footer', 'navigation', 'sidebar', 'menu', 'breadcrumbs', 'pagination', 'legal', 'ads']

        unique_texts = set()

        for tag in content_tags:
            for element in soup.find_all(tag):
                class_list = element.get('class', [])
                id_name = element.get('id', '')
                if not any(excluded in class_list for excluded in excluded_classes) and id_name not in excluded_ids:
                    text_block = re.sub(r'\s+', ' ', element.get_text()).strip()
                    if text_block not in unique_texts:
                        unique_texts.add(text_block)

        text = ' '.join(unique_texts)

        return text


class AIAssistant:
    def __init__(self):
        load_dotenv()
        self.organization_id = os.getenv("OPENAI_ORG_ID")
        self.api_key = os.getenv("OPENAI_KEY")
        self.client = OpenAI(api_key=self.api_key, organization=self.organization_id)

    def parameter_generation(self, function_call: str, question_LLM: str, documents: str, company_name: str, evaluation: bool):
        client = self.client

        try:
            with open("available_functions.json", "r") as json_file:
                available_functions = json.load(json_file)

                if function_call not in available_functions:
                    print("No such function")
                    sys.exit()

                function_to_call = available_functions[function_call]

        except FileNotFoundError:
            print(f"Error: JSON file not found.")
            sys.exit()

        function = function_to_call

        file = client.files.create(
            file=open(documents, 'rb'),
            purpose='assistants'
        )

        # Load Jinja2 template from system_message
        file_loader = FileSystemLoader('.')
        env = Environment(loader=file_loader)
        template = env.get_template('system_message.jinja2')

        # Render the template with your actual values
        system_message = template.render(company_name=company_name)

        parameter_assistant = client.beta.assistants.create(
            instructions=system_message,
            model="gpt-3.5-turbo-1106",
            tools=[{"type": "retrieval"}, {"type": "function", "function": function}],
            file_ids=[file.id],
            name=f"{company_name}_{function_call} parameter assistent",
        )

        parameter_thread = client.beta.threads.create()

        parameter_message = client.beta.threads.messages.create(
            thread_id=parameter_thread.id,
            role="user",
            content=question_LLM,
        )

        #print("bot 1 running ...")
        run = client.beta.threads.runs.create(
            thread_id=parameter_thread.id,
            assistant_id=parameter_assistant.id,
        )

        # Loop until the run status is either "completed" or "requires_action"
        while run.status == "in_progress" or run.status == "queued":
            time.sleep(5)
            run = client.beta.threads.runs.retrieve(
                thread_id=parameter_thread.id,
                run_id=run.id
            )
            #print(run.status)

        if run.status == "requires_action":
            function_result = run.required_action.submit_tool_outputs.tool_calls[0].function
            function_name = function_result.name
            function_args = function_result.arguments
            print(f"1st {company_name} Parameters: {function_args}")
        else:
            pass
                    
        client.beta.threads.runs.cancel(
            run_id=run.id,
            thread_id=parameter_thread.id
        )

        client.beta.threads.delete(
            thread_id=parameter_thread.id
        )

        client.beta.assistants.files.delete(
            file_id=file.id,
            assistant_id=parameter_assistant.id
        )

        client.beta.assistants.delete(
            assistant_id=parameter_assistant.id
        )

        if evaluation:
            # Load Jinja2 template from evaluation_message
            eval_file_loader = FileSystemLoader('.')
            eval_env = Environment(loader=eval_file_loader)
            eval_template = eval_env.get_template('evaluation_message.jinja2') 

            # Render the template with your actual values
            evaluation_message = eval_template.render(company_name=company_name)

            evaluation_assistant = client.beta.assistants.create(
                instructions=evaluation_message,
                model="gpt-3.5-turbo-1106",
                tools=[{"type": "retrieval"}],
                file_ids=[file.id],
                name="evaluation assistent",
            )

            evaluation_thread = client.beta.threads.create()

            eval_message = client.beta.threads.messages.create(
                thread_id=evaluation_thread.id,
                role="user",
                content=f"For each parameter, elaborate whether the assigned score is correct or not. Here are the parameter scores: {function_args}",
            )

            #print("bot 2 running ...")
            run2 = client.beta.threads.runs.create(
                thread_id=evaluation_thread.id,
                assistant_id=evaluation_assistant.id,
            )

            i = 0
            while run2.status not in ["completed", "failed", "requires_action"]:
                if i > 0:
                    time.sleep(10)
                run2 = client.beta.threads.runs.retrieve(
                    thread_id=evaluation_thread.id,
                    run_id=run2.id
                )
                i += 1
                #print(run2.status)

            evaluation_messages = client.beta.threads.messages.list(
                thread_id=evaluation_thread.id
            )

            for message in evaluation_messages:
                if message.role == "assistant":
                    feedback_message = message.content[0].text.value
                    print(f"{company_name} feedback retrieved")
                    #print(feedback_message)


            client.beta.threads.delete(
                thread_id=evaluation_thread.id
            )

            client.beta.assistants.files.delete(
                file_id=file.id,
                assistant_id=evaluation_assistant.id
            )

            client.beta.assistants.delete(
                assistant_id=evaluation_assistant.id
            )

            # Load Jinja2 template from feedback_message
            feedback_file_loader = FileSystemLoader('.')
            feedback_env = Environment(loader=feedback_file_loader)
            feedback_template = feedback_env.get_template('feedback_message.jinja2') 

            # Render the template with your actual values
            feedback_message = feedback_template.render(company_name=company_name)

            parameter_assistant = client.beta.assistants.create(
                instructions=feedback_message,
                model="gpt-3.5-turbo-1106",
                tools=[{"type": "function", "function": function}],
                name=f"{company_name}_{function_call} parameter assistent",
            )

            parameter_thread = client.beta.threads.create()

            parameter_message = client.beta.threads.messages.create(
                thread_id=parameter_thread.id,
                role="user",
                content=f"{question_LLM} Also take into account this feedback: {feedback_message}",
            )

            #print("bot 1 re-running ...")
            run3 = client.beta.threads.runs.create(
                thread_id=parameter_thread.id,
                assistant_id=parameter_assistant.id,
            )

            # Loop until the run status is either "completed" or "requires_action"
            while run3.status == "in_progress" or run3.status == "queued":
                time.sleep(10)
                run3 = client.beta.threads.runs.retrieve(
                    thread_id=parameter_thread.id,
                    run_id=run3.id
                )
                #print(run3.status)

            if run3.status == "requires_action":
                function_result = run3.required_action.submit_tool_outputs.tool_calls[0].function
                function_name = function_result.name
                function_args = function_result.arguments
                print(f"2nd {company_name} Parameters: {function_args}")
            else:
                pass

            client.beta.threads.runs.cancel(
                run_id=run3.id,
                thread_id=parameter_thread.id
            )

            client.beta.threads.delete(
                thread_id=parameter_thread.id
            )

            client.beta.assistants.delete(
                assistant_id=parameter_assistant.id
            )

        client.files.delete(file_id=file.id)

        return function_name, function_args

