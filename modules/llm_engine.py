import os
import json
import threading
from google import genai
from google.genai import types

class LLMEngine:
    def __init__(self, api_key, model_name="gemini-1.5-flash", system_prompt=None):
        self.enabled = False
        if not api_key or "YOUR_API_KEY" in api_key:
            print("[LLMEngine] API key missing. LLM features disabled.")
            return

        try:
            self.client = genai.Client(api_key=api_key)
            self.model_name = model_name
            self.system_prompt = system_prompt
            self.enabled = True
            print(f"[LLMEngine] Initialized with model: {model_name} (using google-genai)")
        except Exception as e:
            print(f"[LLMEngine] Initialization error: {e}")

    def generate_narrative_decision(self, context_json):
        if not self.enabled:
            return None

        prompt = f"Analyze the current social context and decide if you should speak.\nContext: {json.dumps(context_json)}"
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_prompt,
                    response_mime_type="application/json",
                ),
            )
            decision = json.loads(response.text.strip())
            
            # Print reasoning for debugging/logging
            if "reasoning" in decision:
                print(f"[LLM Brain] {decision['reasoning']} (Score: {decision.get('importance', 0)})")
                
            return decision
        except Exception as e:
            print(f"[LLMEngine] Generation error: {e}")
            return None

    def answer_user_query(self, context_json, user_query):
        """
        Force the LLM to directly answer the user's spoken question
        based on the current visual context.
        """
        if not self.enabled:
            return "I am offline."

        # We don't want JSON here, we just want a spoken answer.
        prompt = f"The user just asked you a question. Answer the question naturally in 1 to 2 sentences using the visual data below.\nVisual Context: {json.dumps(context_json)}\nUser Question: '{user_query}'"
        
        try:
            # We use text output for this, not structured JSON.
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_prompt,
                )
            )
            return response.text.strip()
        except Exception as e:
            print(f"[LLMEngine] Q&A Generation error: {e}")
            return "I'm sorry, I'm having trouble thinking."

    def analyze_interactions(self, history_json):
        """
        Perform deeper social reasoning on temporal history.
        """
        if not self.enabled:
            return None

        prompt = f"Analyze these recent social interactions and provide a one-sentence insight:\n{json.dumps(history_json)}"
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            print(f"[LLMEngine] Analysis error: {e}")
            return None
