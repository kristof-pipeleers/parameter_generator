import json
from concurrent.futures import ThreadPoolExecutor
from parameter_generator import WebScraper, AIAssistant
from export_to_csv import transform_to_csv
import os


def replace_company_name(question_template, company_name):
    return question_template.replace("${company_name}", company_name)

def process_company(company_name, questions, ai_assistant, web_scraper):
    collected_data = []

    for question in questions:
        #print(f"company: {company_name}, question: {question}")
        function_call = question["function_call"]
        replaced_question = replace_company_name(question["question"], company_name)
        replaced_llm_question = replace_company_name(question["LLM_question"], company_name)

        temp_file_path = f"scraped_docs_{company_name}_{function_call}.json"

        # Run the existing script with the temporary file
        urls = web_scraper.retrieve_serp_urls(question=replaced_question, num_search_results=10)
        web_scraper.crawl_urls(urls=urls, temp_file_path=temp_file_path)
        function_name, function_args = ai_assistant.parameter_generation(function_call=function_call, question_LLM=replaced_llm_question, documents=temp_file_path, company_name=company_name, evaluation=True)

        os.remove(temp_file_path)
        # Store the collected data in a dictionary
        data_entry = {
            "company_name": company_name,
            "answer": {
                "name": function_name,
                "arguments": function_args
            }
        }

        # Append the data entry to the list
        collected_data.append(data_entry)

    return collected_data

def process_questions_parallel(batch_file, company_names_file):
    # Read company names from file
    with open(company_names_file, "r") as names_file:
        company_names = [line.strip() for line in names_file.readlines()]

    # Read questions from batch file
    with open(batch_file, "r") as batch_file:
        questions = [json.loads(line) for line in batch_file.readlines()]

    ai_assistant = AIAssistant()
    web_scraper = WebScraper()

    collected_data = []

    # Use ThreadPoolExecutor for parallel execution
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit tasks for each group of 4 companies
        futures = [executor.submit(process_company, company_name, questions, ai_assistant, web_scraper) for company_name in company_names]

        # Gather results
        for future in futures:
            collected_data.extend(future.result())

    # Save the collected data to a JSON file
    output_file_path = "collected_data_parallel.json"
    with open(output_file_path, "w") as output_file:
        json.dump(collected_data, output_file, indent=2)

    print(f"Collected data saved to {output_file_path}")

    transform_to_csv(output_file_path)


if __name__ == "__main__":
    batch_file_path = "batch_run_small.jsonl"
    company_names_file_path = "companies.txt"
    process_questions_parallel(batch_file_path, company_names_file_path)
