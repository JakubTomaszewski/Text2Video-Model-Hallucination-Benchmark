import torch    
from transformers import AutoModelForCausalLM, LlamaTokenizer, BitsAndBytesConfig
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
from utils import recur_move_to, collate_fn

class CogVLM:
    def __init__(self, quantization: bool, device: str="cuda", **kwargs) -> None:
        """ Class for the cogvlm model.
        Args:
        device (str): device to run the model
        kwargs: additional arguments
        quantization (bool): whether to quantize the model or not
        """
        self.tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
        self.model = self.load_model(quantization)
        self.device = device
        if quantization:
            self.precision = torch.float16
        else:
            self.precision = torch.bfloat16

    def load_model(self, quantization: bool):
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
    
    def prepare_batch(self, images, query):
        input_list = [self.model.build_conversation_input_ids(
            self.tokenizer, images=[img], query=query, history=[],
            ) for img in images]
        input_batch = collate_fn(input_list, self.tokenizer)
        input_batch = recur_move_to(input_batch, 'cuda', lambda x: isinstance(x, torch.Tensor))
        input_batch = recur_move_to(input_batch, self.precision, lambda x: isinstance(x, torch.Tensor) and torch.is_floating_point(x))
        return input_batch
    
    def run(self, images, query):
        input_batch = self.prepare_batch(images, query)

        gen_kwargs = {"max_length": 2048, "do_sample": False}
        with torch.no_grad():
            outputs = self.model.generate(**input_batch, **gen_kwargs)
            outputs = outputs[:, input_batch['input_ids'].shape[1]:]
            responses = self.tokenizer.batch_decode(outputs)
        return responses