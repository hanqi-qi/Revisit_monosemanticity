# Revisit Monosemanticity from a Feature Decorrelation Perspective
EMNLP24:[Encourage or Inhibit Monosemanticity? Revisit Monosemanticity from a Feature Decorrelation Perspective](https://arxiv.org/pdf/2406.17969)


In this work, we study the mechanistic interpretability in neuron-level, i.e.,Monosemanticity via feature decorrelation(FD) perspective. 
- We firstly provide theoretical justification in section 3.2. 
- We then empirically demonstrate that monosemanticity consistently exhibits a positive correlation with model capacity, in the preference alignment process. 
- We propose DecPO to increase the monosemanticity and further improve DPO via increase reward margin.

### File Organization
```
PlotFigure
|-- Figure2                                          # Contains all files related to plotting Figure 2.
|   |-- weight_statistics_Fig2                       # The folder contains the statistics file for each model.            
    |   |-- GPT-J(6B).npy                            # The folder contains the statistics file for GPT-J.
    |   |-- ...
|   |-- weight_stats.py                              # The file contains the code for plotting.
|   |-- save_weight_statistics.py                    # Save the normalized median ||w_in||^2 b_in for an LLM.
|-- Figure3                                          # Contains all files related to plotting Figure 3.
|   |-- weight_statistics_Fig3                       # The folder contains the statistics file for each model.            
    |   |-- GPT-J(6B).npy                            # The folder contains the statistics file for GPT-J.
    |   |-- ...
|   |-- weight_stats.py                              # The file contains the code for plotting.
|   |-- ComputDiff.py                                # Compute the relative change of normalized median (||w_in||^2 b_in) after an LLM trained with DPO.
|--feature_decorr (figure4 and figure5)
|   |--extract_acts.py
|   |--vis_results.ipynb
|-- requirements.txt                                 # Python environment file.

```
### Monosemanticity is not consistent with model size (Figure2)

For Figure 2, The processed data statistics files for all LLMs, are located in the "weight_statistics_Fig2" folder. Here is the link to download it: [link](https://drive.google.com/file/d/1bC9IKy90gwYbIUWrvjUruqkt9-hJ5YFG/view?usp=drive_link)

**1. Obtain Statistics For each Model**

    cd Figure2
    
    python save_weight_statistics.py --model_name 'GPT2-medium'
    

**2. Plot**

    cd Figure2
    
    python weight_stats.py

### DPO-trained models have larger monosemanticity (Figure3), especially in early layers

For Figure 3, The processed data statistics files for all LLMs, are located in the "weight_statistics_Fig3" folder. Here is the link to download it: [link](https://drive.google.com/file/d/1MwxHvNvbCujgHRcnNIS-9cBJdGkNUWKw/view?usp=drive_link)

**1. Obtain Relative Changes For each Model**

    cd Figure3
    
    python ComputDiff.py --model_name "GPT2-medium" --dpo_path "dpo.pt"


**2. Plot**

    cd Figure3

    python weight_stats.py


## DecPO increases the FD, comparing to DPO (better than base model)

We firstly extract the activations using hook from llama models (base, DPO and DecPO), then calculate the decorrelation (1-cosSimi).
**extract activations**
    cd feature_decorr
    python extract_acts.py
