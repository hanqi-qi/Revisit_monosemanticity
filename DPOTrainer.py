
from trl import DPOTrainer
import torch
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import torch.nn.functional as F
import torch.nn as nn
from models.model_utils import disable_dropout
from models.llama_hook import get_feas_by_hook
from utils.prompt import hf_template_dict, chat_template_dict, hf_split_tag, chat_split_tag
import transformers
from lora_attribute.args import (
    ModelArguments,
    TrainingArguments, 
    LoraArguments, 
    LorraArguments,
)

parser = transformers.HfArgumentParser(
    (ModelArguments, TrainingArguments, LoraArguments, LorraArguments)
)
(
    model_args,
    training_args,
    lora_args,
    lorra_args,
) = parser.parse_args_into_dataclasses()

training_args.model_signature = model_args.model_name_or_path.split("/")[-1]
training_args.prompt_template_dict = chat_template_dict if "chat" in training_args.model_signature else hf_template_dict
training_args.prompt_template = training_args.prompt_template_dict[training_args.dataset_name]
training_args.split_tag = chat_split_tag[training_args.dataset_name] if "chat" in training_args.model_signature else hf_split_tag[training_args.dataset_name]

reference_model = transformers.AutoModelForCausalLM.from_pretrained(
    model_args.model_name_or_path,
    cache_dir=training_args.cache_dir,
    device_map="auto"
)
print("Reference model Loaded!")
disable_dropout(reference_model)
reference_model.eval()
    
class CustomDPOTrainer(DPOTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # Pass all other arguments using **kwargs
        training_args = kwargs["args"]
        self.gamma = 0.2
        self.beta = training_args.beta
        self.reference_model = reference_model
        self.act_layers = [int(layer) for layer in training_args.act_layers.split(",")]
        # self.reference_free = False 
        print(self.act_layers )

    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        ref_pos_logps: torch.FloatTensor,
        ref_neg_logps: torch.FloatTensor,
        reference_free: bool = False,
        ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:       
    
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_pos_logps - ref_neg_logps
        logits = pi_logratios - ref_logratios
        losses = -F.logsigmoid(self.beta * logits)
        chosen_rewards = self.beta * (policy_chosen_logps - ref_pos_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - ref_neg_logps).detach()
        
        return losses, chosen_rewards, rejected_rewards

    def simpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the SimPO loss for a batch of policy model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the SimPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        gamma_logratios = self.gamma / self.beta 
        pi_logratios = pi_logratios.to(self.accelerator.device)
        logits = pi_logratios - gamma_logratios

        if self.loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge']"
            )

        chosen_rewards = self.beta * policy_chosen_logps.to(self.accelerator.device).detach()
        rejected_rewards = self.beta * policy_rejected_logps.to(self.accelerator.device).detach()

        return losses, chosen_rewards, rejected_rewards
    
    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = (
            {
                "labels": concatenated_batch["concatenated_labels"],
                "decoder_input_ids": concatenated_batch.pop("concatenated_decoder_input_ids", None),
            }
            if self.is_encoder_decoder
            else {}
        )

        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
            **model_kwargs,
        ).logits

        all_logps = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=True,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the SimPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}
        
        fea_hooks = get_feas_by_hook(model,target_acts=["mlp.up_proj"],target_layers=self.act_layers)
        
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
        ) = self.concatenated_forward(model, batch)

        # losses, chosen_rewards, rejected_rewards = self.simpo_loss(
        #     policy_chosen_logps,
        #     policy_rejected_logps
        # )
        # if self.reference_free:
        #     ref_pos_logps, ref_neg_logps = 0,0
        # else:
        with torch.no_grad():
            (
            ref_pos_logps,
            ref_neg_logps,
            ref_pos_logits,
            ref_neg_logits,
            ) = self.concatenated_forward(self.reference_model, batch)
        
        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_pos_logps,
            ref_neg_logps,
            # self.reference_free,
        )

        out_feats = []
        for act_nlayer in fea_hooks["mlp.up_proj"]:
        #     # covariant_matrix =
            out_feats.append(torch.abs(act_nlayer.fea[:,-1,:]))
        out_feats_final = torch.stack(out_feats,dim=0)
        
        metrics = {}
        chosen_act = out_feats_final[0,0::2,:]
        sparsity_loss = (torch.matmul(chosen_act,chosen_act.T)-torch.eye(chosen_act.shape[0]).to(chosen_act.device)).mean(dim=0)
        # reject_act = out_feats_final[:,1::2,:]
        # sparsity_loss = training_args.sparse_lambda * torch.sum(chosen_act,dim=-1).mean(dim=0)
        losses += 0.001*sparsity_loss
        
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean().cpu()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().mean().cpu()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().mean().cpu()
        metrics[f"{prefix}sparsity_loss"] = sparsity_loss.detach().mean().cpu()

        return losses.mean(), metrics
    
    # def evaluate(self, ignore_keys=None, sanity_check=False,training_args=training_args, **kwargs):
    #     self.model.eval()
    #     bsz = training_args.per_device_eval_batch_size
    #     reward_types = training_args.reward_types
    #     evaluate_nums = training_args.evaluate_nums
    #     split_tag = training_args.split_tag
    #     torch.cuda.empty_cache()
    #     if sanity_check:
    #         print('Sanity check ...')
    #     metrics = {}
    #     eval_dataset_names = training_args.eval_dataset
    #     for val_set_name in eval_dataset_names:
    #         questions, answers, labels = load_queries(val_set_name,split="valid")
    #         print(f'Evaluating {val_set_name} on {evaluate_nums} samples with {bsz} BSZ...')
    #         with torch.no_grad():
    #             if labels is not None:
    #                 #classification task
    #                 acc = get_logprobs_accuracy(self.model, self.tokenizer, questions, answers, labels, bsz)
    #                 acc_key = 'acc' if val_set_name == 'tqa' else 'acc_norm'
    #                 metrics[f"{val_set_name}_accuracy"] = acc[acc_key]
    #             else:
    #                 querys = questions[:evaluate_nums]
    #                 responses = get_model_responses(self.model, self.tokenizer,querys,val_set_name,training_args)
    #                 print(len(responses))
    #                 step = str(time.time()).split(".")[0]
    #                 metrics = reward_utils.mulreward_evaluate(querys,responses,reward_types,"cuda:0",dataset_name= val_set_name,references=None,verbose=False)
    #                 reward_utils.save_results(result_output_dir, metrics, questions[:len(responses)], responses, reward_types, f"{training_args.model_signature}_{step}_{reward_types[0]}")
    #         wandb.log(metrics)
    #     self.model.train()
    #     print("===Eval results===")
    #     print(metrics)
    #     return metrics