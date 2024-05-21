import json


with open('prompts.json') as json_file:
    data = json.load(json_file)

# Iterate over each prompt in the 'prompts' list
for prompt in data['prompts']:
    # Check if 'people' key exists in the 'object' dictionary
    if 'people' in prompt['object']:
        # Change 'people' to 'person'
        prompt['object']['person'] = prompt['object'].pop('people')
    if 'backpacks' in prompt['object']:
        # Change 'backpacks' to 'backpack'
        prompt['object']['backpack'] = prompt['object'].pop('backpacks')
    if 'ball' in prompt['object']:
        # Change 'ball' to 'sports ball' as the model call it
        prompt['object']['sports ball'] = prompt['object'].pop('ball')
    if 'kitten' in prompt['object']:
        # Change 'kitten' to 'cat' as the model does not distinguish between them
        prompt['object']['cat'] = prompt['object'].pop('kitten')
    if 'kayak' in prompt['object']:
        # Change 'kayak' to 'boat' as the model does not distinguish between boats
        prompt['object']['boat'] = prompt['object'].pop('kayak')
    if 'cardinal' in prompt['object']:
        # Change 'cardinal' to 'bird' as the model does not distinguish between boats
        prompt['object']['bird'] = prompt['object'].pop('cardinal')

# Save the modified JSON data back to the file
with open('prompts_modified.json', 'w') as json_file:
    json.dump(data, json_file, indent=4)
