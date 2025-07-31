import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file (if it exists)
load_dotenv()

# Get the API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Check if we're using an OpenRouter API key (typically starts with 'sk-or')
is_openrouter = api_key.startswith('sk-or')

# Initialize the OpenAI client with appropriate base URL if using OpenRouter
if is_openrouter:
    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )
else:
    client = OpenAI(api_key=api_key)

def generate_palindromic_paragraph(prompt_path="prompts/full_paragraph.txt"):
    with open(prompt_path, "r") as f:
        prompt = f.read()
    
    # Use google/gemma-3n-e2b-it:free model with OpenRouter
    model = "google/gemma-3n-e2b-it:free"
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=250,
        timeout=30
    )
    
    return response.choices[0].message.content.strip()
