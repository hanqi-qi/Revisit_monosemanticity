#!/bin/bash


CUDA_VISIBLE_DEVICES=3 python lora_inference.py \
--model_name_or_path  "/scratch/prj/lmrep/llama2_model/Llama-2-7b-hf" \
--dataset_name 'tatsu-lab/alpaca' \
--reward_types 'hh_rlhf_helpful' \
--user_tag '[INST]' \
--assistant_tag '[/INST]' \
--pos_type 'a helpful' \
--neg_type 'an useless' \
--control_template "Give {type} answer." \
--target_layers "10,12,14,16,18,20" \
--lorra_alpha 5 \
--lorra_beta 0 \
--lora_r 8 \
--lora_alpha 16 \
--lora_dropout 0.05 \
--output_dir ./results/lorra_inference/ \
--overwrite_output_dir \
--num_train_epochs 10 \
--bf16 True \
--per_device_train_batch_size 64 \
--per_device_eval_batch_size 64 \
--gradient_accumulation_steps 1 \
--do_eval \
--evaluation_strategy "steps" \
--eval_steps 100  \
--save_strategy "steps" \
--save_steps 100 \
--learning_rate 3e-4 \
--weight_decay 0. \
--lr_scheduler_type "constant" \
--logging_strategy "steps" \
--logging_steps 100 \
--tf32 True \
--model_max_length 128 \
--q_lora False \
--gradient_checkpointing True \
--report_to none \