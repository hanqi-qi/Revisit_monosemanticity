import matplotlib.pyplot as plt 
import numpy as np 
import os


def plot_normalized_median_norm_bias(models, model_stats):
    plt.figure(figsize=(5,3))
    num = 0

    color_list = [ 'g', 'r', 'b', 'y', 'gray']
    line_list = ['-', '-', '-', '-', '-']
    sp_score_list = list()
    for model in models:
        in_norm = model_stats[model]['in_norm']
        in_bias = model_stats[model]['in_bias']
        n_layers, n_neurons = in_norm.shape 

        sp_score = np.median((in_norm * in_bias), axis=1) / \
            np.max(np.median(np.abs(in_norm * in_bias), axis=1))
        sp_score_list.append(sp_score)
        relative_depth = np.arange(n_layers) / (n_layers - 1)
        plt.plot(relative_depth, sp_score, label=model, color=color_list[num], linestyle=line_list[num])
        num += 1
    plt.legend(ncol=3, loc='upper left',
              title='Model', fontsize=6)
    plt.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    plt.xlabel('relative layer depth',fontsize=10)
    plt.ylabel('normalized median($||W_{in}||_2 b_{in}$)',fontsize=10)
    plt.xlim(-0.005, 1.005)
    plt.tight_layout()
    plt.savefig("./Figure2.pdf") 


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
    models = ['GPT2-medium(355M)', 'GPT2-large(774M)', 'GPT2-xl(1.5B)', 'GPT-neo(2.7B)', 'GPT-J(6B)']
    model_stats = dict()
    for model in models:
        stat = load_weight_statistics(model, "./weight_statistics_Fig2")
        model_stats[model] = stat 
    plot_normalized_median_norm_bias(models, model_stats)