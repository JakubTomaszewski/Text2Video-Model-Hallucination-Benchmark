import os
import torch
import transformers

from .datatypes import Message, Role


class SentenceSimilarityEvaluator:
    def __init__(self,
                 model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
                 device: str = "cuda",
                 prompt: str = None,
                 **kwargs):
        self.device = device
        self.model = self.get_model(model_name, device)
        self.prompt = Message(role=Role.SYSTEM, content=prompt)
        self.terminators = [
            self.model.tokenizer.eos_token_id,
            self.model.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

    def get_model(self, model_name: str, device: str) -> transformers.pipeline:
        model = transformers.pipeline(
            task="text-generation",
            model=model_name,
            device=device,
            model_kwargs={"torch_dtype": torch.bfloat16},
            token=os.environ.get("HF_TOKEN", None)
        )
        return model

    def _create_conversation(self, sentence_1: str, sentence_2: str) -> list[Message]:
        user_message_content = f"First sentence: '{sentence_1}'. Second sentence: '{sentence_2}'."
        messages = [
            self.prompt,
            Message(role=Role.USER, content=user_message_content),
        ]

        conversation = self.model.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return conversation

    def evaluate(self, sentence_1: str, sentence_2: str) -> str:
        """Evaluate whether two sentences are similar.

        Args:
            sentence_1 (str): sentence 1
            sentence_2 (str): sentence 2

        Returns:
            str: decision on whether the two sentences are similar
        """
        conversation = self._create_conversation(sentence_1, sentence_2)

        outputs = self.model(
            conversation,
            max_new_tokens=256,
            eos_token_id=self.terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        print("Sentence similarity output:", outputs[0]["generated_text"])
        return outputs[0]["generated_text"][len(self.prompt):]
