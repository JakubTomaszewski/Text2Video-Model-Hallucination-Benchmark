def get_indefinite_article(word):
    vowels = "aeiouAEIOU"
    first_letter = word[0]
    if first_letter in vowels:
        return "an"
    else:
        return "a"
    
def choose_template(num, object_, color):
    article = get_indefinite_article(object_)
    if num==1:
        return f"Please briefly answer: What is the color of the {object_} in the image?"
    elif num==2:
        return f"Please briefly answer: Find the color of the {object_} in the image."
    elif num==3:
        return f"Please briefly answer: If there is {article} {object_} in the image, what is its color?"
    elif num==4:
        # Color confusion
        return f"Are there any {color} objects in the image other than {article} {object_}. Let's think step by step."
    
def recur_move_to(item, tgt, criterion_func):
    if criterion_func(item):
        device_copy = item.to(tgt)
        return device_copy
    elif isinstance(item, list):
        return [recur_move_to(v, tgt, criterion_func) for v in item]
    elif isinstance(item, tuple):
        return tuple([recur_move_to(v, tgt, criterion_func) for v in item])
    elif isinstance(item, dict):
        return {k: recur_move_to(v, tgt, criterion_func) for k, v in item.items()}
    else:
        return item

def collate_fn(features, tokenizer) -> dict:
    images = [feature.pop('images') for feature in features]
    tokenizer.padding_side = 'left'
    padded_features = tokenizer.pad(features)
    inputs = {**padded_features, 'images': images}
    return inputs