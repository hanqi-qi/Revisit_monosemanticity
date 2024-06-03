from .demo import DemoProbInferenceForStyle
task_mapper = {
    'demo': DemoProbInferenceForStyle,
    'sentiment': DemoProbInferenceForStyle,
    'toxicity': DemoProbInferenceForStyle,
    'simplicity': DemoProbInferenceForStyle,
    'helpfulness': DemoProbInferenceForStyle,
    'stack_qa': DemoProbInferenceForStyle,
    'hh_rlhf': DemoProbInferenceForStyle,
    'cog_reframe_positive_paired_data': DemoProbInferenceForStyle,
    "wiki2_nontoxic_paired_data":DemoProbInferenceForStyle,
    'sycophancy_ab_paired_data': DemoProbInferenceForStyle,
    "sycophancy_philpapers2020_paired_data":DemoProbInferenceForStyle,
    "3H_paired_data":DemoProbInferenceForStyle,
}


def load_task(name):
    if name not in task_mapper.keys():
        raise ValueError(f"Unrecognized dataset `{name}`")

    return task_mapper[name]
