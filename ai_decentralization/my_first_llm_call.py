# Traceback (most recent call last):
#   File "<stdin>", line 3, in <module>
#   File "<frozen os>", line 685, in __getitem__
# KeyError: 'HF_TOKEN'

import os
from openai import OpenAI

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

completion = client.chat.completions.create(
    model="moonshotai/Kimi-K2-Instruct",
    messages=[
        {
            "role": "user",
            "content": "Generate a list of 10 interesting facts about space."
        }
    ],
)

print(completion.choices[0].message)