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

        print(f"{len(results)} are retrieved")
        return [item['link'] for item in results]

    def crawl_urls(self, urls: List[str]):
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

            print(url)
            docs.append({"url": url, "content": text})

        with open('scraped_docs.txt', 'w', encoding='utf-8') as txt_file:
            for doc in docs:
                txt_file.write(str(doc) + '\n')

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

    def parameter_generation(self, function_call: str, question_LLM: str, data: str):
        client = self.client

        print(f"function to call: {function_call}")

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

        system_message = '''Je bent een chatbot die de gebruiker zal helpen bij het invullen van verschillende bedrijfsparameters. Je zal altijd moeten antwoorden in de vorm van een JSON formaat zoals aangegeven in de vraag. Voor het beantwoorden van de vragen, maak gebruik van de voorgaande context informatie. Als u het antwoord niet weet, geef dan aan dat u het niet weet zonder een speculatief antwoord te geven. Vermeld de bron van uw informatie alleen als u deze heeft gebruikt bij het opstellen van uw antwoord.'''

        with open(data, "r", encoding="utf-8", errors="replace") as file:
            file_content = file.read()
        max_instructions_length = 28000
        file_content = file_content[:max_instructions_length]

        file = client.files.create(
            file=open(data, 'rb'),
            purpose='assistants'
        )

        parameter_assistant = client.beta.assistants.create(
            instructions=system_message,
            model="gpt-3.5-turbo-1106",
            tools=[{"type": "retrieval"}, {"type": "function", "function": function}],
            file_ids=[file.id],
        )

        parameter_thread = client.beta.threads.create()

        parameter_message = client.beta.threads.messages.create(
            thread_id=parameter_thread.id,
            role="user",
            content=question_LLM
        )

        time.sleep(5)

        print("bot 1 running ...")
        run = client.beta.threads.runs.create(
            thread_id=parameter_thread.id,
            assistant_id=parameter_assistant.id
        )

        i = 0
        while run.status not in ["completed", "failed", "requires_action"]:
            if i > 0:
                time.sleep(10)
            run = client.beta.threads.runs.retrieve(
                thread_id=parameter_thread.id,
                run_id=run.id
            )
            i += 1
            print(run.status)

        tools_to_call = run.required_action.submit_tool_outputs.tool_calls
        print(len(tools_to_call))
        #print(tools_to_call)

        tool_output_array = []

        for tool in tools_to_call:
            tool_id = tool.id
            function_name = tool.function.name
            function_args = tool.function.arguments
            print(f"Parameters: {function_args}")

        client.beta.threads.runs.cancel(
            run_id=run.id,
            thread_id=parameter_thread.id
        )
        #print(run.status)


        system_message2 = '''System:
        You are an AI assistant tasked with evaluating a company's parameters based on a provided context. The parameters are either scored on a scale from 0 to 2, where 0 denotes 'not applicable,' 1 signifies 'moderately applicable,' and 2 represents 'highly applicable.' Alternatively, they may be expressed as TRUE or FALSE, with TRUE indicating the parameter best describing the company.

        Your role is to assess the accuracy of the parameter scores in the ANSWER within the given CONTEXT.

        User:
        You will be provided with a CONTEXT and an ANSWER regarding that CONTEXT. The ANSWER consists of various company parameters, each assigned a score between 0 and 2 OR TRUE or FALSE to indicate their relevance to the company. Your task is to determine whether the scores assigned to these parameters in the ANSWER align with the CONTEXT.

        Thoroughly read the provided information to understand the CONTEXT and then select the appropriate answer label from the three options. Additionally, provide a brief explanation for your evaluation. For each parameter, indicate whether the assigned score is correct or not.

        Note that the ANSWER is generated by a computer system and may contain certain symbols, which should not negatively impact your evaluation.

        Independent Examples:

        ## Example Task #1 Input:
        {"CONTEXT": "Our partner company HungA distributes them mainly in Asian countries. The Schwalbe brand is, however, a name which is more well-known all over the world.", "ANSWER": "{"answer": {"role": "assistant","function_call": {"name": "get_company_activities","arguments": "{\n "Distribution": 0}"}}}"}

        ## Example Task #1 Output:
        The company Schwalbe is involved in the distribution sector, so a score of 0 is likely inaccurate. Consider the CONTEXT to assign a more appropriate value to this parameter.

        ## Example Task #2 Input:
        {"CONTEXT": "Green Compound (rubber compund exclusively made of renewable materials)\nWASTE MINIMIZATION\nResource-efficient packaging\n100 % recyclable packaging\nEMPLOYEES\nComprehensive support & assistance\nLong-term employment (further education, cycle to work scheme, etc.)\nNEW HEADQUARTERS\n70 % of materials used are circular\nENERGY CONCEPT", "ANSWER": {"role": "assistant","function_call": {"name": "get_company_activities","arguments": "{\n "Waste": 3}"}}}"}

        ## Example Task #2 Output:
        The company Shimano is indeed involved in waste recovery.

        ## Example Task #3 Input:
        {"CONTEXT": "The logistics center for global distribution is inaugurated. It is also the corporate headquarters and workplace for 100 employees when the Schwalbe brand was set up in Germany, there were just seven employees.", "ANSWER": {"role": "assistant","function_call": {"name": "get_company_category","arguments": "{\n "KMO": True}"}}}"}

        ## Example Task #2 Output:
        The company Schwalbe is not a KMO but a corporate. Hence, the paramater KMO set to True is likely inaccurate.

        Reminder: Keep your feedback concise for each parameter.
        '''

        '''
        evaluation_assistant = client.beta.assistants.create(
            instructions=system_message2,
            model="gpt-3.5-turbo-1106",
            tools=[{"type": "retrieval"}],
            file_ids=[file.id]
        )

        evaluation_thread = client.beta.threads.create()

        evaluation_message = client.beta.threads.messages.create(
            thread_id=evaluation_thread.id,
            role="user",
            content=f"For each parameter, indicate whether the assigned score is correct or not. Here are the parameter scores: {function_args}"
        )

        time.sleep(10)

        print("bot 2 running ...")
        run2 = client.beta.threads.runs.create(
            thread_id=evaluation_thread.id,
            assistant_id=evaluation_assistant.id,
            instructions=f'"CONTEXT": "{file_content}", "ANSWER": {function_args}'
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
            print(run2.status)

        evaluation_messages = client.beta.threads.messages.list(
            thread_id=evaluation_thread.id
        )

        feedback_message = ""
        for message in evaluation_messages:
            if message.role == "assistant":
                feedback_message = message.content[0].text.value
                print(feedback_message)

        client.beta.assistants.delete(
            assistant_id=evaluation_assistant.id
        )

        parameter_message = client.beta.threads.messages.create(
            thread_id=parameter_thread.id,
            role="user",
            content=question_LLM
        )

        time.sleep(5)

        print("bot 1 re-running ...")
        run3 = client.beta.threads.runs.create(
            thread_id=parameter_thread.id,
            assistant_id=parameter_assistant.id,
            instructions=f"Take this feedback into account: {feedback_message}"
        )

        i = 0
        while run3.status not in ["completed", "failed", "requires_action"]:
            if i > 0:
                time.sleep(10)
            run3 = client.beta.threads.runs.retrieve(
                thread_id=parameter_thread.id,
                run_id=run3.id
            )
            i += 1
            print(run3.status)

        tools_to_call = run3.required_action.submit_tool_outputs.tool_calls
        print(len(tools_to_call))
        #print(tools_to_call)

        tool_output_array = []

        for tool in tools_to_call:
            tool_id = tool.id
            function_name = tool.function.name
            function_args = tool.function.arguments
            print(f"Parameters: {function_args}")

        client.beta.threads.runs.cancel(
            run_id=run3.id,
            thread_id=parameter_thread.id
        )
        #print(run3.status) 
        '''

        client.beta.assistants.files.delete(
            file_id=file.id,
            assistant_id=parameter_assistant.id
        )

        client.files.delete(
            file_id=file.id
        )

        client.beta.assistants.delete(
            assistant_id=parameter_assistant.id
        )

        return function_name, function_args


# Usage
web_scraper = WebScraper()
ai_assistant = AIAssistant()

'''urls = web_scraper.retrieve_serp_urls(question="shimano company activities", num_search_results=10)
web_scraper.crawl_urls(urls=urls)

ai_assistant.parameter_generation(response_message="get_company_activities",
                                  question_LLM="Provide a score for following business activities for the company Shimano: "
                                               "Bio-economy, Mining, Component, Part, Assembly, Distribution, Design, "
                                               "Energy recovery, and Waste.",
                                  data="scraped_docs.txt")'''
