#!/bin/bash

python evaluate_from_ckpt.py \
--model_name_or_path  "/scratch/prj/lmrep/llama2_model/Llama-2-7b-chat-hf" \
--dataset_name 'challenge_toxicity' \
--eval_dataset 'challenge_toxicity' \
--reward_types 'alignment' \
--user_tag '[INST]' \
--assistant_tag 'Output:' \
--pos_type 'a positive' \
--neg_type 'a negative' \
--control_template "Give {type} answer." \
--target_layers "10,12,14,16,18,20" \
--lorra_alpha 5 \
--lorra_beta 0 \
--lora_r 8 \
--lora_alpha 16 \
--lora_dropout 0.05 \
--output_dir ./results/multi_dpo_Wosparse_WoType_WoRefer_gate_down/ \
--overwrite_output_dir \
--num_train_epochs 50 \
--bf16 True \
--evaluate_nums 200 \
--reference_free False \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 32 \
--gradient_accumulation_steps 1 \
--evaluation_strategy "steps" \
--eval_steps 500  \
--save_strategy "steps" \
--save_steps 200 \
--learning_rate 3e-4 \
--weight_decay 0. \
--lr_scheduler_type "constant" \
--logging_strategy "steps" \
--logging_steps 50 \
--tf32 True \
--model_max_length 128 \
--q_lora False \
--gradient_checkpointing True \
--report_to "wandb" \


#dataset_name: 'wiki2_nontoxic_paired_data'/hh_rlhf_helpful_paired_data/cog_reframe_positive_paired_data/challenge_toxicity/sycophancy_ab_paired_data