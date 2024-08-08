from groq import Groq
import json
from openai import OpenAI
import os

class SecondLayer():
    def __init__(self):
        self.client = OpenAI(api_key= os.environ.get("OPENAI_API_KEY"))
       
    def predict(self,question,code):
        with open("context.json","r",encoding= "utf-16") as f:
            data = json.load(f)
        classes = data[0][str(code)]["classes"]
        guideline = data[0][str(code)]["guideline"]
        history = data[0][str(code)]["history"]
        response = self.client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": f"""
You are an expert AI assistant specializing in categorizing user messages related to banking and financial services.Your task is to analyze the user's message and categorize it into one of the following classes:
<classes>
{classes}
</classes>

To accurately categorize the messages, follow these guidelines:
<guidelines>
{guideline}
</guidelines>
The messages will be in Vietnamese, so you must be proficient in understanding colloquial Vietnamese expressions and banking-related terminology. Be prepared to interpret various ways users might phrase their inquiries or concerns.

Respond only with the appropriate category name, without any additional explanation or commentary. If the user's message doesn't clearly fit into any of the specified categories, choose the most appropriate one based on the context provided.

Your categorization should be accurate, consistent, and aligned with the examples provided in the training data. Always strive for the most precise classification possible given the information in the user's message."""
            },
            *history,
            {
                "role": "user",
                "content": question
            }
        ],
        temperature=0.0,
        max_tokens=1024,
        top_p= 0.5,
        stop=None,
        stream=False
    )
        return response.choices[0].message.content