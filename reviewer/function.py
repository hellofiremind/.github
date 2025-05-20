import os
import json
import boto3
import logging
import requests
import argparse
from pathlib import Path
from string import Template
from bs4 import BeautifulSoup
from atlassian import Confluence
from requests.auth import HTTPBasicAuth

from model_information import MODEL_IDS

parser = argparse.ArgumentParser(description="")
parser.add_argument("--filename", action="store", dest="filename")
parser.add_argument(
    "--review-type",
    action="store",
    dest="reviewType",
    choices=["infrastructure", "frontend", "backend", "python"],
)

args = parser.parse_args()

REGION_NAME = os.getenv("REGION_NAME", "us-east-1")
CONFLUENCE_URL = os.getenv("CONFLUENCE_URL", "https://firemind.atlassian.net/wiki")
EMAIL = os.getenv("EMAIL", "james.staples@firemind.com")
CONFLUENCE_API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN")

session = boto3.Session(region_name=REGION_NAME)
bedrock = session.client("bedrock-runtime")

logger_level = os.getenv("LOGGER_LEVEL", "INFO").upper()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def get_confluence_page_content(page_id, confluence_url, email, api_token):
    url = f"{confluence_url}/rest/api/content/{page_id}?expand=body.storage"
    response = requests.get(
        url,
        auth=HTTPBasicAuth(email, api_token),
        headers={"Accept": "application/json"},
    )
    if response.status_code == 200:
        data = response.json()
        page_title = data.get("title")
        page_content = (
            data.get("body", {}).get("storage", {}).get("value", "No content found.")
        )
        return page_title, page_content
    else:
        error_message = f"Error: {response.status_code} - {response.text}"
        return None, error_message



def extract_headers_and_content(content):
    content_dict = {}

    soup = BeautifulSoup(content, "html.parser")

    current_header = None
    current_content = []

    for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "p"]):
        if tag.name in ["h1", "h2", "h3", "h4", "h5"]:
            if current_header:
                content_dict[current_header] = " ".join(current_content).strip()

            current_header = tag.get_text().strip()
            current_content = []

        elif tag.name == "p" and current_header:
            current_content.append(tag.get_text().strip())

    if current_header:
        content_dict[current_header] = " ".join(current_content).strip()

    return content_dict


def invoke_bedrock(model_id, messages, system_prompt, client=bedrock):

    temperature = 0.2
    top_p = 0

    inference_config = {"temperature": temperature, "maxTokens": 4000, "topP": top_p}

    response = client.converse(
        modelId=model_id,
        messages=messages,
        system=system_prompt,
        inferenceConfig=inference_config,
    )

    input_tokens = float(response["usage"]["inputTokens"])
    output_tokens = float(response["usage"]["outputTokens"])
    total_tokens = float(response["usage"]["totalTokens"])
    latency = float(response["metrics"]["latencyMs"])

    bedrock_metadata = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "latency": latency,
    }
    return response, bedrock_metadata


def format_message(role, content):
    return [{"role": role, "content": [{"text": content}]}]


def format_prompt(prompt):
    return [
        {
            "text": prompt,
        }
    ]


def generate_output(model_id, system_prompt, messages, client=bedrock):

    response, _ = invoke_bedrock(
        messages=messages, model_id=model_id, system_prompt=system_prompt
    )
    final_response = response["output"]["message"]["content"][0]["text"]

    return final_response


output_format = """
APIs: 
1. Use ESM JavaScript, aiming for the latest stable node version: Pass 
2. If you can, use LLRT (Optional): Fail
...
Data Pipelines: 
1. Prefer Python, aiming for the latest stable python version: Pass
...
"""
agent_system_prompt_templates = {
    "infrastructure": {
        "template": Template(
            """
        You are a code review agent at Firemind, an AWS Advanced Partner specializing in AI and ML solutions.  
        Your task is to review Terraform code submitted by Firemind developers for alignment with Firemind's Terraform technology standards.
        Instructions:
        - Only apply the relevant guidelines from the provided standards. If a section does not apply to the code being reviewed (e.g., no outputs) ignore it completely.
        - Treat "Examples", and "Notes" as context, not mandatory checklist items.
        - For each relevant guideline section:
            - State the guideline name as a small header.
            - Summarize what that section expects.
            - Provide a status of either "Complete" or "Incomplete" for the reviewed file. Alternatively this could be **n/a** if the guideline cannot be judged.
            - Give a brief justification for the status, referencing only what was found or missing.
        - Do not include any guideline that was not applicable.
        - At the start of the report, include a heading with the reviewed filename, e.g., Code Review for infrastructure/s3.tf.

        Firemind's Terraform Standards:  
        $tech_choice_doc

        Ensure your feedback is pragmatic, relevant, and focused on real-world maintainability and code quality.
        Return the review in this format:

        ## Code Review for filename.tf

        ### 1. [Guideline Title]
        Status: Complete/Incomplete
        Summary: [One-sentence summary of what the guideline is about]
        Notes: [Why it's complete or what is missing]

        ### 2. [Guideline Title]
        ...
        
        ### Summary
        [Short overview of how the file performs againt the guidelines] 
        """
        ),
        "pageID": "3871703059",
    },
    "python": {
        "template": Template(
            """
        You are a code review agent at Firemind, an AWS Advanced Partner specializing in AI and ML solutions.
        Your task is to review Python code submitted by Firemind developers for alignment with Firemind's Python technology standards.

        Instructions:
        - Only apply the relevant guidelines from the provided standards. If a section does not apply to the code being reviewed (e.g., it covers classes and the code has none), ignore it completely.
        - Treat "Red Flags", "Examples", and "Notes" as context, not mandatory checklist items.
        - For each relevant guideline section:
            - State the guideline name as a small header.
            - Summarize what that section expects.
            - Provide a status of either "Complete" or "Incomplete" for the reviewed file.
            - Give a brief justification for the status, referencing only what was found or missing.
        - Do not include any guideline that was not applicable.
        - At the start of the report, include a heading with the reviewed filename, e.g., Code Review for app1/function.py.
        
        Fireminds Python Standards:
        $tech_choice_doc

        Ensure your feedback is pragmatic, relevant, and focused on real-world maintainability and code quality.
        Return the review in this format:
        
        ## Code Review for filename.tf

        ### 1. [Guideline Title]
        Status: Complete/Incomplete
        Summary: [One-sentence summary of what the guideline is about]
        Notes: [Why it's complete or what is missing]

        ### 2. [Guideline Title]
        ...
        
        ### Summary
        [Short overview of how the file performs againt the guidelines] 
        """
        ),
        "pageID": "3871703045",
    },
}


def handle_code_review_request(page_id, filepath, system_prompt_template):
    context = Path(filepath).read_text()
    title, content = get_confluence_page_content(
        page_id, CONFLUENCE_URL, EMAIL, CONFLUENCE_API_TOKEN
    )
    tech_choice_doc = extract_headers_and_content(content)
    system_prompt = system_prompt_template.substitute(
        output_format=output_format, tech_choice_doc=tech_choice_doc
    )
    formatted_message = format_message("user", context)
    formatted_prompt = format_prompt(system_prompt)
    response = generate_output(
        MODEL_IDS["nova-pro"]["id"], formatted_prompt, formatted_message
    )
    return response


def agent(filename, reviewType):
    try:
        response = handle_code_review_request(
            reviewType["pageID"], filename, reviewType["template"]
        )
        print(response)
        return response

    except Exception as e:
        logger.error(f"Error when invoking agent: {e}")


agent(args.filename, agent_system_prompt_templates[args.reviewType])
