python training_reward_model.py 
    --model_name $PATH_TO_TULU_CKPT \
    --dataset_name "/scratch/prj/cllm/datasets/psoups/data/rm_training/P1A.json" \
    --eval_dataset_name $EVAL_DATASET_NAME \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size 2 \
    --num_train_epochs 1 \
    --wandb_project $WANDB_PROJECT_NAME \
    --wandb_run_name $WANDB_RUN_NAME