# The code for plotting figure 2 and figure 3

## File Organization
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
|-- requirements.txt                                 # Python environment file.

```
## Figure 2

For Figure 2, The processed data statistics files for all LLMs, are located in the "weight_statistics_Fig2" folder. Here is the link to download it: [link](https://drive.google.com/file/d/1bC9IKy90gwYbIUWrvjUruqkt9-hJ5YFG/view?usp=drive_link)

**1. Obtain Statistics For each Model**

    cd PlotFigure/Figure2
    
    python save_weight_statistics.py --model_name 'GPT2-medium'
    

**2. Plot**

    cd PlotFigure/Figure2
    
    python weight_stats.py

## Figure 3 

For Figure 3, The processed data statistics files for all LLMs, are located in the "weight_statistics_Fig3" folder. Here is the link to download it: [link](https://drive.google.com/file/d/1MwxHvNvbCujgHRcnNIS-9cBJdGkNUWKw/view?usp=drive_link)

**1. Obtain Relative Changes For each Model**

    cd PlotFigure/Figure3
    
    python ComputDiff.py --model_name "GPT2-medium" --dpo_path "dpo.pt"


**2. Plot**

    cd PlotFigure/Figure3

    python weight_stats.py


## Figure 4 and Figure 5 (Feature Decorrelation)

We firstly extract the activations using hook from llama models, then calculate the decorrelation (1-cosSimi).
