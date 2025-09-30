import torch
from tqdm import tqdm
import ast
import wandb
import numpy as np
import os
import math
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import to_edge_index

import yaml 
from yaml import SafeLoader
from torch.utils.data import random_split


from data.load import load_data
from data.sampling import ego_graphs_sampler # , collect_subgraphs
from utils.peft import create_peft_config
from utils.args import Arguments
from models.encoder import GCN_Encoder, SAGE_Encoder, GIN_Encoder, MLP_Encoder, GAT_Encoder, PMLP_Encoder, GCNII_Encoder

torch.serialization.add_safe_globals([torch.utils.data.dataset.Subset])


def get_hidden_states(config):
    path = f'./llm_cache/{config.dataset}/layers'
    if not os.path.exists(os.path.join(path, 'layer_attr.pt')):
        raise FileNotFoundError(f'No cache found! Please use `python cache.py --dataset {config.dataset}` to generate it.')

    else:
        layers_hid = torch.load(os.path.join(path, 'layer_attr.pt'), weights_only=False)

    xs = layers_hid
    return xs  # 미리 저장된 LLM hidden들

def get_dataloader_ddxplus(graphs, config):
    kwargs = {'batch_size': 4, 'num_workers': 6, 'persistent_workers': True} # 256

    if config.dataset in ['ddxplus'] and os.path.exists(f'./subgraphs/{config.dataset}/khop-1/train.pt') and os.path.exists(f'./subgraphs/{config.dataset}/khop-1/val.pt') and os.path.exists(f'./subgraphs/{config.dataset}/khop-1/test.pt'):
        print('using cache of subgraphs')
        train_graphs = torch.load(f'./subgraphs/{config.dataset}/khop-1/train.pt', weights_only=False)
        val_graphs = torch.load(f'./subgraphs/{config.dataset}/khop-1/val.pt', weights_only=False)
        test_graphs = torch.load(f'./subgraphs/{config.dataset}/khop-1/test.pt', weights_only=False)
    else:
        n_total = len(graphs)
        n_train = int(0.6 * n_total)
        n_val   = int(0.2 * n_total)
        n_test  = n_total - n_train - n_val
        train_graphs, val_graphs, test_graphs = random_split(
            graphs, [n_train, n_val, n_test]
        )
        os.makedirs(f'./subgraphs/{config.dataset}/khop-1')
        torch.save(train_graphs, f'./subgraphs/{config.dataset}/khop-1/train.pt')
        torch.save(val_graphs, f'./subgraphs/{config.dataset}/khop-1/val.pt')
        torch.save(test_graphs, f'./subgraphs/{config.dataset}/khop-1/test.pt')

    train_loader = DataLoader(train_graphs, shuffle=True, **kwargs)
    val_loader = DataLoader(val_graphs, **kwargs)
    test_loader = DataLoader(test_graphs, **kwargs)
    return train_loader, val_loader, test_loader

def efficient_train_eval(train_loader, val_loader, test_loader, xs, model_list, prog_list,  alpha_list, exit_list, optimizer):
    patience = config.patience
    best_acc = 0
    best_test_from_val = 0
    best_state_list = []
    cnt = 0
    
    criterion = torch.nn.CrossEntropyLoss()
    # criterion = torch.nn.BCEWithLogitsLoss()
    os.makedirs("./checkpoints", exist_ok=True)
    ckpt_path = f"./checkpoints/{config.dataset}_best.pt"

    for epoch in tqdm(range(config.epochs)):
        epoch_loss = 0
        step = 0
        for batch in train_loader:
            batch = batch.to(device)
            data_list = batch.to_data_list()   # 배치 안의 각 subgraph 분리

            for data in data_list:
                optimizer.zero_grad()
                last = None
                total_loss = 0
                for i, m in enumerate(model_list):
                    m.train()
                    prog_list[i].train()
                    exit_list[i].train()

                    idx_tensor = torch.tensor(data.original_idx, dtype=torch.long, device=device)

                    if i == 0:
                        out = m(prog_list[i](xs[i][idx_tensor]), data.edge_index)
                    else:
                        a = torch.sigmoid(alpha_list[i]/T)
                        x = prog_list[i](xs[i][idx_tensor]) * a + last * (1-a)
                        out = m(x, data.edge_index)

                    last = out
                    hid_out = last[data.disease_idx]
                    hid_logits = exit_list[i](hid_out).squeeze(-1)
                    # hid_logits = exit_list[i](hid_out).squeeze()   # [num_candidates]
                    gold_idx = torch.argmax(data.y[data.disease_idx]).item()
                    # labels = data.y[data.disease_idx].float()      # [num_candidates]
                    labels = torch.tensor(gold_idx, dtype=torch.long, device=device)
                    loss = criterion(hid_logits.view(1, -1), labels.view(1))
                    # loss = criterion(hid_logits, labels)
                    total_loss += loss

                total_loss.backward(retain_graph=True)
                optimizer.step()
                
        epoch_loss += total_loss.item()
        step += 1

        avg_loss = epoch_loss / max(1, step)
        val_acc = efficient_eval(val_loader, xs, model_list, prog_list,  alpha_list, exit_list)
        test_acc = efficient_eval(test_loader, xs, model_list, prog_list,  alpha_list, exit_list)
                # 🔥 wandb 로깅
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_acc": val_acc,
            "test_acc": test_acc
        })
        if val_acc > best_acc:
            best_acc = val_acc
            cnt = 0
            best_test_from_val = test_acc

            checkpoint = {
                f"model_{i}": model_list[i].state_dict() for i in range(len(model_list))
            }
            checkpoint.update({
                f"prog_{i}": prog_list[i].state_dict() for i in range(len(prog_list))
            })
            checkpoint.update({
                f"exit_{i}": exit_list[i].state_dict() for i in range(len(exit_list))
            })
            checkpoint.update({
                f"alpha_{i}": alpha_list[i].data for i in range(len(alpha_list))
            })

            torch.save(checkpoint, ckpt_path)
            print(f"✅ Best model updated & saved at {ckpt_path} (val_acc={val_acc:.4f})")

        else:
            cnt += 1

    return best_test_from_val


def efficient_eval(test_loader, xs, model_list, prog_list, alpha_list, exit_list):
    correct = 0
    total_cnt = 0
    table = wandb.Table(columns=["pred", "gold", "is_correct"])
    for batch in test_loader:
        batch = batch.to(device)
        data_list = batch.to_data_list()

        for data in data_list:
            labels = data.y[data.disease_idx].float()
            print("labels", labels)
            total_cnt += len(data.disease_idx)

            last = None
            results = torch.zeros(len(data.disease_idx), device=device)

            for i, m in enumerate(model_list):
                m.eval()
                prog_list[i].eval()
                exit_list[i].eval()

                idx_tensor = torch.tensor(data.original_idx, dtype=torch.long, device=device)

                if i == 0:
                    out = m(prog_list[i](xs[i][idx_tensor]), data.edge_index)
                    hid_out = out[data.disease_idx]
                else:
                    a = torch.sigmoid(alpha_list[i] / T)
                    x = prog_list[i](xs[i][idx_tensor]) * a + last * (1-a)
                    out = m(x, data.edge_index)
                    hid_out = out[data.disease_idx]

                last = out
                hid_logits = exit_list[i](hid_out).squeeze(-1) 
                # hid_logits = exit_list[i](hid_out).squeeze()
                # hid_prob = torch.sigmoid(hid_logits)
                results += hid_logits
            print("result label: \n", results)
            pred_idx = results.argmax().item()
            # gold = labels.argmax(dim=0).item()
            gold_idx = torch.argmax(data.y[data.disease_idx]).item()
            idx_to_entity = ast.literal_eval(data.idx_to_entity)
            
            print(f"예측: {idx_to_entity[pred_idx]}  |  정답: {idx_to_entity[gold_idx]}")
            is_correct = int(pred_idx == gold_idx)
            table.add_data(idx_to_entity[pred_idx], idx_to_entity[gold_idx], is_correct)
            correct += is_correct

    acc = correct / total_cnt if total_cnt > 0 else 0
    wandb.log({"predictions": table, "eval_acc": acc})
    return acc

if __name__ == '__main__':
    config = Arguments().parse_args()
    args = yaml.load(open(config.config), Loader=SafeLoader)
    # combine args and config
    for k, v in args.items():
        config.__setattr__(k, v)
    print(config)

    # ✅ wandb init
    wandb.init(
        project="ddxplus-engine",   # 프로젝트 이름 (원하는 걸로 변경 가능)
        name=f"{config.dataset}-seed-exp",  # 실험 이름
        config=vars(config)  # yaml + CLI args 그대로 로깅
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    xs = get_hidden_states(config)
    xs = [x for x in xs]
    print(f"total layer 수 : {len(xs)}. model layer + embedding layer. model is bert.")
    
    # for seed in range(5):
    # load data
    graphs, text_id_dict = load_data(config.dataset, seed=config.seeds[0])        
    train_loader, val_loader, test_loader = get_dataloader_ddxplus(graphs, config)
    
    r=config.r # used for dimensionality reduction
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
    model_list = [encoders[config.encoder](k, config.layer_num, hidden, k, activation=config.activation, norm=config.norm, last_activation=(l !=len(layer_select)-1), dropout=config.dropout).to(device) for l in layer_select]
    prog_list = [torch.nn.Sequential(torch.nn.Linear(input_dim, k), torch.nn.LayerNorm(k), torch.nn.ReLU(), torch.nn.Linear(k,k)).to(device) for l in layer_select]
    alpha_list = [torch.nn.Parameter(torch.tensor(0.0), requires_grad=True) for l in layer_select]
    exit_list = [torch.nn.Linear(k, 1).to(device) for l in layer_select]  # binary classifier 단순히 layer select 개수 만큼 바이너리 output layer가 정의된 것
    classifier = torch.nn.Linear(k, 1).to(device)
    T=config.T
    lr = config.lr
    weight_decay = config.weight_decay
    
    params = []
    xs_list = []
    for i, l in enumerate(layer_select):
        params.append({'params': model_list[i].parameters(), 'lr': lr, 'weight_decay': weight_decay}) 
        params.append({'params': prog_list[i].parameters(), 'lr': lr, 'weight_decay': weight_decay}) 
        params.append({'params': alpha_list[i], 'lr': lr, 'weight_decay': weight_decay})
        params.append({'params': exit_list[i].parameters(), 'lr': lr, 'weight_decay': weight_decay})
        xs_list.append(xs[l])
    params.append({'params': classifier.parameters(), 'lr': lr, 'weight_decay': weight_decay})
    
    optimizer = torch.optim.AdamW(params)
    
    # ENGINE w/ caching

    acc = efficient_train_eval(train_loader, val_loader, test_loader, xs_list, model_list, prog_list, alpha_list, exit_list, optimizer)

    print(f"# final_acc: {acc*100:.2f}")
    
### CUDA_VISIBLE_DEVICES=3 python3 -m main --config ./configs/ddxplus/engine.yaml