import os
import logging
from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai

logger = logging.getLogger(__name__)

class AIModels:
    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None
        self.gemini_model = None

    def setup_openai(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OpenAI API key not found")
            raise ValueError("OpenAI API key not set")
        self.openai_client = OpenAI(api_key=api_key)
        logger.info("OpenAI client set up successfully")

    def setup_anthropic(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.error("Anthropic API key not found")
            raise ValueError("Anthropic API key not set")
        self.anthropic_client = Anthropic(api_key=api_key)
        logger.info("Anthropic client set up successfully")

    def setup_gemini(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.error("Google API key not found")
            raise ValueError("Google API key not set")
        genai.configure(api_key=api_key)
        self.gemini_model = genai.GenerativeModel("gemini-1.5-pro")
        logger.info("Google Gemini model set up successfully")

    def setup_all(self):
        self.setup_openai()
        self.setup_anthropic()
        self.setup_gemini()

    def process_openai_prompt(self, prompt):
        try:
            response = self.openai_client.chat.completions.create(
                model="chatgpt-4o-latest",
                messages=[{"role": "system", "content": prompt["prompt"]}],
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error processing OpenAI prompt: {str(e)}")
            raise

    def process_anthropic_prompt(self, prompt):
        try:
            response = self.anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                temperature=0,
                messages=[{"role": "user", "content": prompt["prompt"]}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error processing Anthropic prompt: {str(e)}")
            raise

    def process_gemini_prompt(self, prompt):
        try:
            response = self.gemini_model.generate_content(prompt["prompt"])
            return response.text
        except Exception as e:
            logger.error(f"Error processing Gemini prompt: {str(e)}")
            logger.error(f"Response: {response}")