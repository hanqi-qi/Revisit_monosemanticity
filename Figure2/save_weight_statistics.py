import argparse
import os
import torch
import numpy as np
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM


def load_model(model_name="gpt2-small", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_name == 'GPT2-medium':
        path = 'gpt2-medium'
        model = HookedTransformer.from_pretrained(path, device='cpu')
    elif model_name == 'GPT2-large':
        path = 'gpt2-large'
        model = HookedTransformer.from_pretrained(path, device='cpu')
    elif model_name == 'GPT2-xl':
        path = 'gpt2-xl'
        model = HookedTransformer.from_pretrained(path, device='cpu')
    elif model_name == "GPT-J":
        model = HookedTransformer.from_pretrained("EleutherAI/gpt-j-6B", device='cpu')
    elif model_name == "GPT-neo-2.7B":
        model = HookedTransformer.from_pretrained("EleutherAI/gpt-neo-2.7B", device='cpu')
    
    model.eval()
    torch.set_grad_enabled(False)
    if model.cfg.device != device:
        try:
            model.to(device)
        except RuntimeError:
            print(
                f"WARNING: model is too large to fit on {device}. Falling back to CPU")
            model.to('cpu')

    return model


def compute_and_save_weight_statistics(model_name):   
    model = load_model(model_name, device='cpu')
    in_norm = model.W_in.norm(dim=1).numpy()
    in_bias = model.b_in.numpy()
    out_norm = model.W_out.norm(dim=-1).numpy()
    out_bias = model.b_out.numpy()
    cos = torch.nn.CosineSimilarity()(model.W_in, torch.swapaxes(model.W_out, 1, 2))

    n_layers, n_neurons = in_norm.shape
    statistics = np.zeros((5, n_layers, n_neurons))
    statistics[0] = in_norm
    statistics[1] = in_bias
    statistics[2] = out_norm
    statistics[3, :, :len(out_bias[0])] = out_bias
    statistics[4] = cos

    save_dir = 'weight_statistics_Fig2'

    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, f'{model_name}.npy')
    np.save(save_file, statistics)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model_name', type=str, default="GPT2-medium")
    args = parser.parse_args()
    compute_and_save_weight_statistics(args.model_name)
