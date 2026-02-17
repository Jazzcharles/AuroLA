# from .vast import VAST
from .qwen_omni import Qwen2_50OmniRetForConditionalGeneration
from .qwen_omni_finetune import Qwen2_50OmniRetFinetuneForConditionalGeneration

# from .qwen_omni_atm import Qwen2_50OmniRetForConditionalGenerationATM
# from .qwen_omni_finetune_atm import Qwen2_50OmniRetFinetuneForConditionalGenerationATM 
# from .qwen3 import Qwen3RetForCausalLM
# from .qwen3_omni import Qwen3OmniRetForConditionalGeneration
# from .qwen3_omni_finetune import Qwen3OmniRetFinetuneForConditionalGeneration
model_registry = {
    # 'vast':VAST,
    'qwen':Qwen2_50OmniRetForConditionalGeneration,
    # 'qwen-atm':Qwen2_50OmniRetForConditionalGenerationATM,
    # 'qwen': Qwen2_50OmniRetFinetuneForConditionalGeneration,
    # 'qwen3': Qwen3OmniRetForConditionalGeneration, ### qwen3-omni-30b-a3b ###
}
