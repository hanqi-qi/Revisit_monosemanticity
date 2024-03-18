python3 contrast_inference.py \
--dataset sentiment \
--model_type llama-2 \
--model_size "7b" \
--start_id 0 \
--end_id 100 \
--reward_types sentiment \
--generate_woICV True \