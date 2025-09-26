# inference.py
import torch
import numpy as np
import os
import yaml
from yaml import SafeLoader
from tqdm import tqdm
from torch_geometric.loader import DataLoader

from data.load import load_data
from utils.args import Arguments
from models.encoder import GCN_Encoder, SAGE_Encoder, GAT_Encoder, MLP_Encoder

# hidden states 캐시 로드
def get_hidden_states(config):
    path = f'./llm_cache/{config.dataset}/layers'
    if not os.path.exists(os.path.join(path, 'layer_attr.pt')):
        raise FileNotFoundError(f'No cache found! Please run `python cache.py --dataset {config.dataset}` first.')
    return torch.load(os.path.join(path, 'layer_attr.pt'))

# dataloader (train/val은 불필요 → test만 사용)
def get_test_loader(graphs, config):
    n_total = len(graphs)
    n_train = int(0.6 * n_total)
    n_val   = int(0.2 * n_total)
    n_test  = n_total - n_train - n_val
    _, _, test_graphs = torch.utils.data.random_split(graphs, [n_train, n_val, n_test])
    kwargs = {'batch_size': 4, 'num_workers': 6, 'persistent_workers': True}
    return DataLoader(test_graphs, **kwargs)

@torch.no_grad()
def inference(test_loader, xs, model_list, prog_list, alpha_list, exit_list, T, device):
    model_list = [m.to(device).eval() for m in model_list]
    prog_list = [p.to(device).eval() for p in prog_list]
    exit_list = [e.to(device).eval() for e in exit_list]

    correct, total_cnt = 0, 0
    predictions = []

    for data in tqdm(test_loader, desc="Inference"):
        data = data.to(device)
        disease_idx = data.disease_idx
        labels = data.y[disease_idx].float()   # [num_candidates]
        total_cnt += 1   # 그래프 단위로 카운트

        last = None
        for i, m in enumerate(model_list):
            if i == 0:
                out = m(prog_list[i]((xs[i][data.original_idx.cpu()]).to(device)), data.edge_index)
            else:
                a = torch.sigmoid(alpha_list[i] / T)
                x = prog_list[i]((xs[i][data.original_idx.cpu()]).to(device)) * a + last * (1 - a)
                out = m(x, data.edge_index)
            last = out

        hid_out = last[disease_idx]
        hid_logits = exit_list[-1](hid_out).squeeze()   # [num_candidates]
        hid_prob = torch.sigmoid(hid_logits)

        pred = hid_prob.argmax(dim=0).item()
        gold = labels.argmax(dim=0).item()
        predictions.append((data.id, pred, gold))

        if pred == gold:
            correct += 1

    acc = correct / total_cnt if total_cnt > 0 else 0
    return acc, predictions


if __name__ == '__main__':
    config = Arguments().parse_args()
    args = yaml.load(open(config.config), Loader=SafeLoader)
    for k, v in args.items():
        setattr(config, k, v)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    xs = [x for x in get_hidden_states(config)]

    # 데이터 로드
    graphs, _ = load_data(config.dataset, seed=config.seeds[0])
    test_loader = get_test_loader(graphs, config)

    # 모델 초기화
    r=config.r
    input_dim=config.input_dim
    k = int(input_dim/r)
    hidden = config.hidden_size
    layer_select = config.layer_select
    encoders = {
        'GCN_Encoder': GCN_Encoder,
        'GAT_Encoder': GAT_Encoder,
        'SAGE_Encoder': SAGE_Encoder,
        'MLP_Encoder': MLP_Encoder,
    }
    model_list = [encoders[config.encoder](k, config.layer_num, hidden, k,
                                           activation=config.activation, norm=config.norm,
                                           last_activation=(l != len(layer_select)-1),
                                           dropout=config.dropout).to(device) for l in layer_select]
    prog_list = [torch.nn.Sequential(
                    torch.nn.Linear(input_dim, k),
                    torch.nn.LayerNorm(k),
                    torch.nn.ReLU(),
                    torch.nn.Linear(k,k)).to(device) for l in layer_select]
    alpha_list = [torch.nn.Parameter(torch.tensor(0.0), requires_grad=True) for l in layer_select]
    exit_list = [torch.nn.Linear(k*2, 1).to(device) for l in layer_select]

    T=config.T

    # 학습된 weight 로드
    ckpt_path = f'./checkpoints/{config.dataset}/best_model.pt'
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    for obj, state in zip(model_list+prog_list+exit_list, checkpoint["model_states"]):
        obj.load_state_dict(state)

    # Inference
    acc, predictions = inference(test_loader, [xs[l] for l in layer_select],
                                 model_list, prog_list, alpha_list, exit_list, T, device)

    print(f"[Inference] Test Accuracy: {acc*100:.2f}%")
    for pid, pred, gold in predictions[:20]:  # 앞 20개만 출력
        print(f"Case {pid} → Pred: {pred}, Gold: {gold}")
