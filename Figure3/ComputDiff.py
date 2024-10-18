import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from fancy_einsum import einsum
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer
from load import load_model
import pickle
from transformers import AutoModelForCausalLM

torch.set_grad_enabled(False)


def load_hooked(model_name, weights_path):
    hf_model = AutoModelForCausalLM.from_pretrained(model_name,low_cpu_mem_usage=True)
    print(weights_path)
    _weights = torch.load(weights_path, map_location=torch.device("cpu"))[
    "state"
    ]
    hf_model.load_state_dict(_weights)
    model = HookedTransformer.from_pretrained(model_name, device='cpu', hf_model=hf_model)
    return model


def get_svd(model, modeldpo):
    vec_ori = []
    bias_ori = []
    vec_dpo = []
    bias_dpo = []
    index = []
    for layer in range(model.cfg.n_layers):
        mlp_in_ori = model.blocks[layer].mlp.W_in.detach().cpu()
        mlp_in_bias_ori = model.blocks[layer].mlp.b_in.detach().cpu()
        mlp_in_dpo = modeldpo.blocks[layer].mlp.W_in.detach().cpu()
        mlp_in_bias_dpo = modeldpo.blocks[layer].mlp.b_in.detach().cpu()
        cos_sims = F.cosine_similarity(
            mlp_in_dpo, mlp_in_ori, dim=0
        )
        AbsDifference = torch.abs(mlp_in_ori - mlp_in_dpo)
        AbsDifference = torch.sum(AbsDifference, dim=0)
        _topk = AbsDifference.topk(k=100)
        _idxs = [x.item() for x in _topk.indices] 
        idx = torch.LongTensor(_idxs)
        index.append(idx)
        selected_slices_ori = torch.index_select(mlp_in_ori, 1, idx)
        selected_slices_dpo = torch.index_select(mlp_in_dpo, 1, idx)
        biasTerm_ori = torch.index_select(mlp_in_bias_ori, 0, idx)
        biasTerm_dpo = torch.index_select(mlp_in_bias_dpo, 0, idx)
        vec_ori.append(selected_slices_ori)
        vec_dpo.append(selected_slices_dpo)
        bias_ori.append(biasTerm_ori)
        bias_dpo.append(biasTerm_dpo)
    vec_ori = torch.stack(vec_ori, dim=0).norm(dim=1)
    vec_dpo = torch.stack(vec_dpo, dim=0).norm(dim=1)
    bias_ori = torch.stack(bias_ori, dim=0)
    bias_dpo = torch.stack(bias_dpo, dim=0)
    return vec_ori, vec_dpo, bias_ori, bias_dpo


def plot_normalized_median_norm_bias(models, model_stats, ax=None):
    plt.figure(figsize=(5,3))
    num = 0
    sp_score_list = list()
    for model in models:
        in_norm = model_stats[model]['in_norm']
        in_bias = model_stats[model]['in_bias']
        sp_score = np.median((in_norm * in_bias), axis=1) / \
            np.max(np.median(np.abs(in_norm * in_bias), axis=1))
        sp_score_list.append(sp_score)
        num += 1
    score = list()
    for i in range(0, len(sp_score_list[0]), 1):
        score.append((sp_score_list[1][i]-sp_score_list[0][i])/abs(sp_score_list[0][i]))
        print("Layer ", str(i+1), " ", str((sp_score_list[1][i]-sp_score_list[0][i])/abs(sp_score_list[0][i])))
    print(max(score))
    pickle.dump(score, open("./gpt2-large.pkl", 'wb'))


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
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dpo_path', type=str, default="dpo.pt")
    parser.add_argument('--model_name', type=str, default="gpt2-medium")
    args = parser.parse_args()


    modeldpo = load_hooked("gpt2-large", args.dpo_path)
    model = HookedTransformer.from_pretrained(args.model_name)
    
    vec_ori, vec_dpo, bias_ori, bias_dpo = get_svd(model, modeldpo)

    ORI = dict()
    ORI['in_norm'] = vec_ori
    ORI['in_bias'] = bias_ori
    DPO = dict()
    DPO['in_norm'] = vec_dpo
    DPO['in_bias'] = bias_dpo
    models = [args.model_name, args.model_name+"-dpo"]
    model_stats = dict()
    model_stats[args.model_name] = ORI
    model_stats[args.model_name+"-dpo"] = DPO
    plot_normalized_median_norm_bias(models, model_stats)







