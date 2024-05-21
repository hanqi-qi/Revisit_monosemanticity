for dataset_name in "cog_reframe_positive_paired_data"
do
    for train_schema in "sft"
    do
        for act_layers in "2"
        do
            bash ./scripts/train_sft.sh $dataset_name $train_schema $act_layers >TRL_sparisty_mlpUpTraniable_${train_schema}_${dataset_name}_${act_layers}.out 2>&1
        done
    done
done