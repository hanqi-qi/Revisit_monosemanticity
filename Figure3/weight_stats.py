import matplotlib.pyplot as plt 
import numpy as np 
import os
import pickle


def plot_normalized_median_norm_bias(models, model_stats, ax=None):
    plt.figure(figsize=(5,3))
    num = 0
    
    color_list = [ 'g', 'r', 'b', 'y', 'gray']
    line_list = ['-', '-', '-', '-', '-']
    sp_score_list = list()
    for model in models:
        score = model_stats[model]
        n_layers = len(score)
        score = [x * 100 for x in score]
        sp_score_list.append(score)
        relative_depth = np.arange(n_layers) / (n_layers - 1)
        plt.plot(relative_depth, score, label=model, color=color_list[num], linestyle=line_list[num])
        num += 1
    plt.legend(ncol=2, loc='lower left',
              title='Model', fontsize=6)
    plt.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    plt.xlabel('relative layer depth',fontsize=10)
    plt.ylabel('Relative improvement (%)',fontsize=10)
    plt.xlim(-0.005, 1.005)
    plt.tight_layout()
    plt.savefig("./Figure3.pdf") 


def load_weight_statistics(model_name, save_dir=None):
    if save_dir is None:
        save_dir = os.path.join(
            os.environ.get('RESULTS_DIR', 'results'),
            'weight_statistics'
        )
    stats = np.load(os.path.join(save_dir, f'{model_name}.npy'))
    _, _, n_neurons = stats.shape
    return {
        'in_norm': stats[0],
        'in_bias': stats[1],
        'out_norm': stats[2],
        'out_bias': stats[3, :, :n_neurons//4],
        'cos': stats[4]
    }

if __name__ == "__main__":
    models = 'GPT2'
    models = ['GPT2-medium(355M)', 'GPT2-large(774M)', 'GPT2-xl(1.5B)', 'GPT-neo(2.7B)']
    model_stats = dict()
    for model in models:
        path = "./weight_statistics_Fig3/" + model + ".pkl"
        score = pickle.load(open(path, "rb"))
        model_stats[model] = score
    plot_normalized_median_norm_bias(models, model_stats)