# Shallow Company Recommender

This script generates company recommendations using Large Language Models (LLMs) through OpenRouter's API. It processes a CSV file containing company information and generates N recommended companies for each input company.

## Features

- Supports multiple LLM models (GPT-4, Claude 3.5, Perplexity, Gemini)
- Configurable number of recommendations per company
- Structured JSON output for each recommendation

## Prerequisites

- Python 3.9+
- OpenRouter API key

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with your OpenRouter API key:
   ```plaintext
   OPENROUTER_API_KEY=your_api_key_here
   ```

## Input Data Format

The script expects a CSV file with company information. The CSV should be placed in a `data` directory and should contain at minimum:
- `name`: Company name
- Other company information fields that will be used for context

Example structure:
```csv
name,location,industry,description
Company A,New York,Manufacturing,Description here
Company B,California,Technology,Description here
```

## Usage

1. Place your input CSV file in the `data` directory.

2. Run the script:
   ```bash
   python shallow_recommender.py
   ```

By default, the script:
- Uses the Perplexity model
- Generates 50 recommendations per company
- Reads from `data/wmcompanies.csv`
- Outputs to `wmcompanies_with_recs.csv`

## Output

The script generates a new CSV file with the original data plus N new columns:
- `rec_0` through `rec_N-1`: Each containing a dictionary with:
  - `name`: Recommended company name
  - `state`: State where the recommended company is based

## Configuration

You can modify these variables in the script:
- `model`: Choose between 'cgpt', 'claude', 'perplexity', or 'gemini'
- `N`: Number of recommendations per company
- `data_path`: Input file path

## Error Handling

The script includes:
- Retry mechanism for failed API calls
- JSON parsing error handling
- Data storage error handling
- Progress tracking with tqdm

## YOUR TASK

- Run the script with the Perplexity, Claude and cgpt models
- Generate 50 recommendations per company and model
- Store the results in either a new csv file or a new column in the existing csv file per model
- Write a function that ranks the recommendation based on the agreement accross models (i.e., if the same 'name' appears in 
    more than 1 model, that company should be higher in the list than the ones that appear in less models)
- Output a final list with the aggregated recommenations from all models, ranked by agreement

NOTE: We are more interested in creating the pipeline than on the final list itself, so focus on writing modular code with clear input and output 
variables in each function.

ADDITIONAL NOTE: Your api key has a limit of 300 requests per day, so user them strategically (maybe save the llm results to memory the first time you get them and develop the ranking function on these saved copies, so as to not call for results every time you run the script).
