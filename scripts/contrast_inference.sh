python3 contrast_inference.py \
--dataset toxicity \
--model_type llama-2 \
--model_size "7b" \
--start_id 5 \
--end_id 1000 \
--reward_types toxicity \
--generate_woICV True \