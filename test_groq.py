import os
from dotenv import load_dotenv
from groq import Groq

print("Loading .env...")

load_dotenv()

key = os.getenv("GROQ_API_KEY")

print("Key found:", key is not None)

client = Groq(api_key=key)

print("Calling API...")

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {
            "role": "user",
            "content": "Say hello in one sentence."
        }
    ]
)

print("Finished!")

print(response.choices[0].message.content)