import torch
from transformers import AutoModelForCausalLM, LlamaTokenizer, BitsAndBytesConfig
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch

class ColorIdentifier:
    def __init__(self, quantization: bool, device: str="cuda", **kwargs) -> None:
        """Identify the color of the objects in the video frames.
        Args:
        """
        self.tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
        self.model = self.load_model(quantization)

    def load_model(self, quantization):
        if not quantization:
            with init_empty_weights():
                model = AutoModelForCausalLM.from_pretrained(
                    'THUDM/cogvlm-chat-hf',
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                )
            device_map = infer_auto_device_map(model, max_memory={0:'15GiB','cpu':'35GiB'}, no_split_module_classes='CogVLMDecoderLayer')
            path_to_index = '/root/.cache/huggingface/hub/models--THUDM--cogvlm-chat-hf/snapshots/e29dc3ba206d524bf8efbfc60d80fc4556ab0e3c/model.safetensors.index.json'
            model = load_checkpoint_and_dispatch(
                model,
                path_to_index,   # typical, '~/.cache/huggingface/hub/models--THUDM--cogvlm-chat-hf/snapshots/balabala'
                device_map=device_map,
            )
            model = model.eval()
        else:
            bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
            )
            max_memory_mapping = {0: "15GB", 'cpu': "35GB"}
            model = AutoModelForCausalLM.from_pretrained(
                'THUDM/cogvlm-chat-hf',
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                quantization_config=bnb_config,
                max_memory=max_memory_mapping
            ).eval()
        return model
    
    def choose_template(self, num, object_, color):
        article = self.get_indefinite_article(object_)
        if num==1:
            return f"Please briefly answer: What is the color of the {object_} in the image?"
        elif num==2:
            return f"Please briefly answer: Find the color of the {object_} in the image."
        elif num==3:
            return f"Please briefly answer: If there is {article} {object_} in the image, what is its color?"
        elif num==4:
            # Color confusion
            return f"Are there any {color} objects in the image other than {article} {object_}. Let's think step by step."
        
    def get_indefinite_article(self, word):
        vowels = "aeiouAEIOU"
        first_letter = word[0]
        if first_letter in vowels:
            return "an"
        else:
            return "a"

