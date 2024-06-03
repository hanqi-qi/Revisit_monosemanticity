from typing import Optional, Dict, Sequence
from dataclasses import dataclass, field
import transformers
import typing

@dataclass
class LorraArguments:
    user_tag: str = field(metadata={"help": "User tag for chat models (eg: `USER:` or `[INST]`)"})
    assistant_tag: str = field(metadata={"help": "Assistant tag for chat models (eg: `ASSISTANT:` or `[\INST]`)"})
    pos_type: str = field(metadata={"help": "Concept/Function to be optimized towards (eg: 'a truthful')"})
    neg_type: str = field(metadata={"help": "vice versa of pos_type (eg: 'an untruthful')"})
    target_layers: str = field(metadata={"help": "Layers for Representation. Layers are seperate by `,` eg: `10,12,14,16,18,20` "})
    control_template: str = field(metadata={"help": "Control template for Representation setting (eg: Give a {type} answer)"})
    lorra_alpha: float = field(default=5, metadata={"help": "vice versa of pos_type (eg: 'an untruthful')"}) # LoRRA Hyperparameters
    lorra_beta: float = field(default=0, metadata={"help": "vice versa of pos_type (eg: 'an untruthful')"}) # LoRRA Hyperparameters
    max_res_len: int = field(default=64, metadata={"help": "truncated length for getting generated ouputs from lorra pos/neg exampels"}) # LoRRA Hyperparameters

@dataclass
class LoraArguments:
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: typing.List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "mlp.up_proj"]#set all target modules for LoRA
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-2-7b-chat-hf")
    adapter_name_or_path: str = field (
        default=None, metadata={"help": "Adapater name"}
    )
    use_lora: bool = field(
        default=False, metadata={"help": "Use LoRA (default: False)"}
    )
    
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")

    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    prompt_max_len: int = field(
        default=128,
        metadata={"help": "Maximum sequence length for prompt"},
    )
    response_max_len: int = field(
        default=128,
        metadata={"help": "Maximum sequence length for prompt"},
    )
    grouped_to_max_length: bool = field (
        default=False, metadata={"help": "Group to chunks of max length for pretraining"}
    )
    train_schema: str = field(  default="dpo", metadata={"help": "dpo or sft"} )
    
    evaluate_nums: int = field (
        default=200, metadata={"help": "Number of evaluation samples"}
    )
    dataset_name: str = field(
        default = "toxicity_pair",
        metadata={"help": "Dataset name"}
    )
    
    eval_dataset: typing.List[str] = field(
        default_factory=lambda: ["cog_reframe_positive_paired_data"]
    )
    
    reward_types: typing.List[str] = field(
        default_factory=lambda: ["toxicity"]
    )
    
    policy_paths: str = field(
        default = None ,
        metadata={"help": "local paths saving the trained policy models"}
    )

    kl_gamma: float = field(
        default = 0.1,
        metadata={"help": "KL annealing gamma"}
    )
    
    beta: float = field(
        default = 0.1,#0.5
        metadata={"help": "Beta for reward shaping"}
    )
    
    sparse_lambda: float = field(
        default = 0.0001,
        metadata={"help": "Sparsity lambda"}
    )
    
    reference_free: bool = field(
        default = False,
        metadata={"help": "Reference free reward"}
    )
    
    act_layers: str = field(
        default="31",
        metadata={"help": "layers applied iwth sparisty"}
        
    )
    
    use_label: str = field(
        default = "False",
        metadata={"help": "use labelled data for Classifier"}
    )

    learning_rate: float = field(
        default=5e-5, 
        metadata={"help": "The initial learning rate for Adam."}
    )