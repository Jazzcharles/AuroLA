from typing import Tuple

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, AutoModelForCausalLM

from . import register_loader
from .base import BaseModelLoader
from model.qwen_omni import Qwen2_50OmniRetForConditionalGeneration
from model.qwen_omni_finetune import Qwen2_50OmniRetFinetuneForConditionalGeneration
# from model.qwen2_vl import Qwen2VLRetForConditionalGeneration
# from model.qwen2_vl_finetune import Qwen2VLRetFinetuneForConditionalGeneration


@register_loader("qwen2_5-omni-3b")
class Qwen2_5Omni3BModelLoader(BaseModelLoader):
    def load(self, load_model: bool = True, pretrain=True) -> Tuple[AutoModelForCausalLM, AutoTokenizer, None]:
        if load_model and pretrain:
            model = Qwen2_50OmniRetForConditionalGeneration.from_pretrained(
                self.model_local_path, 
                **self.loading_kwargs,
            ) 
        elif load_model and not pretrain:
            model = Qwen2_50OmniRetFinetuneForConditionalGeneration.from_pretrained(
                self.model_local_path, 
                **self.loading_kwargs,
            ) 

        processor = AutoProcessor.from_pretrained(self.model_local_path)
        tokenizer = processor.tokenizer 

        self.add_embed_token(tokenizer, model)

        return model, tokenizer, processor 

    def add_embed_token(self, tokenizer, model, emb_token="<emb>"):
        emb_tokens = [emb_token]
        
        # Check if the token is already in the tokenizer
        if emb_token not in tokenizer.get_vocab():
            print('<emb> token not found in tokenizer, adding it...')
            num_new_tokens = tokenizer.add_tokens(emb_tokens)
            assert len(emb_tokens) == num_new_tokens
            model.resize_token_embeddings(len(tokenizer))
        else:
            print('<emb> token already exists in tokenizer')
        
        emb_token_ids = tokenizer.convert_tokens_to_ids(emb_tokens)
        model.config.emb_token_ids = emb_token_ids

