import os
from dotenv import load_dotenv
import pandas as pd
from typing import Union, Dict
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
import json


# Load environment variables
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

# Available models
MODELS = {
    "cgpt": "openai/gpt-4o-2024-11-20",
    "claude": "anthropic/claude-3.5-sonnet",
    "perplexity": "perplexity/llama-3.1-sonar-large-128k-online",
    "gemini": "google/gemini-pro-1.5"
}


def get_shallow_recs(model_name: str, company: dict, N: int = 50, max_retries: int = 5) -> Union[Dict, None]:
    """
    Generate a list of N recomended companies for a given company.
    Args:
        model_name: Name of the LLM model to use (choose between cgpt, claude, perplexity)
        company: Dictionary containing company information for the prompt
        N: Number of recommendations to generate
        max_retries: Maximum number of retry attempts if parsing fails
        
    Returns:
        Union[Dict, None]: Parsed dictionary containing RAG data if successful, None otherwise
    """
    # Initialize LangChain ChatOpenAI model
    llm = ChatOpenAI(base_url="https://openrouter.ai/api/v1",
                     api_key=api_key,
                     model=model_name, 
                     temperature=0, 
                     max_retries=max_retries)
    
    # Build the system and human prompts
    system_prompt = "You are an expert assistant that provides structured JSON responses."

    human_prompt = f"""
    As an expert consultant in the waste management and transport industry, you've been tasked
    with doing some market research on target company {company.get('name', 'Unknown')}. You have the following information about this target company:
    INFORMATION: {json.dumps(company, indent=2).replace('{', '{{').replace('}', '}}')}
    Your task is to identify {N} companies that are most likely current or past customers of the target company or would be a 
    good fit for the target company's services.

    Please respond with a JSON array of exactly {N} objects. Each object should have the following fields:
    - "name": A string representing the name of the recommended company.
    - "state": A string representing the state where the company is based.

    Here is an example of the required format:
    [
        {{"name": "Company A", "state": "California"}},
        {{"name": "Company B", "state": "Texas"}}
    ]

    IMPORTANT: 
    - Only return valid JSON, without any extra text, explanations, or formatting like markdown.
    - Ensure the JSON is valid and can be parsed programmatically.
    """

    # Create ChatPromptTemplate
    prompt_template = ChatPromptTemplate([
        ("system", system_prompt),
        ("human", "{user_input}")
    ])

    # Set up JSON parser
    parser = JsonOutputParser()
    chain = prompt_template | llm | parser

    attempts = 0
    while attempts < max_retries:
        attempts += 1
        try:
            # Run the chain
            completion = chain.invoke({"user_input": human_prompt})

            # If successful, return the parsed dictionary
            return completion
        except json.JSONDecodeError as e:
            print(f"Attempt {attempts}: Failed to decode JSON. Error: {e}. Retrying...")
        except Exception as e:
            print(f"Attempt {attempts}: An unexpected error occurred: {e}. Retrying...")

    print(f"All {max_retries} attempts failed to get valid RAG data for company: {company['name']}")
    return None


if __name__ == "__main__":
    data_path = "data/wmcompanies_with_rag.csv"
    model = "perplexity"
    N = 50

    companies = pd.read_csv(data_path)[:10]
    model = MODELS[model]
    
    # Create N columns for the recommendations
    for i in range(N):
        companies[f"rec_{i}"] = None

    # Process the companies
    for idx, row in tqdm(companies.iterrows(), total=len(companies)):
        # Convert the entire row to a dictionary
        company = row.to_dict()
        recs = get_shallow_recs(model, company, N)
        
        if recs:
            # Store each recommendation dictionary in its corresponding column
            for i, rec in enumerate(recs[:N]):  # Ensure we only use N recommendations
                try:
                    companies.loc[idx, f"rec_{i}"] = str(rec)  # Convert dict to string
                except Exception as e:
                    print(f"Error storing recommendation {i} for company {company['name']}: {e}")

    companies.to_csv(f"{data_path.split('.')[0].split('_')[0]}_with_recs.csv", index=False)