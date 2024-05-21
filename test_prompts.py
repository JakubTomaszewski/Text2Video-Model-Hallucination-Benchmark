import json

with open('prompts.json', 'r') as f:
    prompt_details = json.load(f)
    prompts = [prompt['sentence'].strip(" .\n") for prompt in prompt_details['prompts']]


for prompt in prompts:
    print(prompt)
