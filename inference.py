import torch
import ast
import yaml
from yaml import SafeLoader
from torch_geometric.data import Data

from data.load import load_data
from utils.args import Arguments
from models.encoder import GCN_Encoder, SAGE_Encoder, GIN_Encoder, MLP_Encoder, GAT_Encoder

@torch.no_grad()
def inference(data, xs, model_list, prog_list, alpha_list, exit_list, device, T):
    """
    단일 그래프 데이터 (torch_geometric.data.Data) 입력 -> 예측 반환
    """
    data = data.to(device)
    labels = data.y[data.disease_idx].float()
    results = torch.zeros(len(data.disease_idx), device=device)
    last = None
    hid_logits = None
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
        results += hid_logits
    idx_to_entity = ast.literal_eval(data.idx_to_entity)
    
    hid_prob = torch.sigmoid(hid_logits)
    pred_idx = results.argmax().item()
    gold_idx = torch.argmax(data.y[data.disease_idx]).item()

    return {
        "pred": idx_to_entity[pred_idx],
        "gold": idx_to_entity[gold_idx],
        "probs": hid_prob.detach().cpu().numpy()
    }

if __name__ == "__main__":
    # 1. Config 불러오기
    config = Arguments().parse_args()
    args = yaml.load(open(config.config), Loader=SafeLoader)
    for k, v in args.items():
        config.__setattr__(k, v)
    print(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Hidden states 불러오기
    from main import get_hidden_states
    xs = get_hidden_states(config)
    xs = [x.to(device) for x in xs]

    # 3. 모델 초기화 (학습 시와 동일하게)
    encoders = {
        'GCN_Encoder': GCN_Encoder, 
        'GAT_Encoder': GAT_Encoder, 
        'SAGE_Encoder': SAGE_Encoder, 
        'MLP_Encoder': MLP_Encoder,
    }
    r = config.r
    input_dim = config.input_dim
    k = int(input_dim / r)
    hidden = config.hidden_size
    layer_select = config.layer_select

    model_list = [encoders[config.encoder](k, config.layer_num, hidden, k,
                                           activation=config.activation,
                                           norm=config.norm,
                                           last_activation=(l != len(layer_select)-1),
                                           dropout=config.dropout).to(device) 
                  for l in layer_select]
    prog_list = [torch.nn.Sequential(
                    torch.nn.Linear(input_dim, k),
                    torch.nn.LayerNorm(k),
                    torch.nn.ReLU(),
                    torch.nn.Linear(k, k)
                 ).to(device) for l in layer_select]
    alpha_list = [torch.nn.Parameter(torch.tensor(0.0, device=device), requires_grad=True) for _ in layer_select]
    exit_list = [torch.nn.Linear(k, 1).to(device) for _ in layer_select]

    T = config.T

    # 4. 학습된 checkpoint 불러오기
    checkpoint = torch.load("./checkpoints/ddxplus_best.pt", map_location=device)
    for i, m in enumerate(model_list):
        m.load_state_dict(checkpoint[f"model_{i}"])
        prog_list[i].load_state_dict(checkpoint[f"prog_{i}"])
        exit_list[i].load_state_dict(checkpoint[f"exit_{i}"])
        alpha_list[i].data = checkpoint[f"alpha_{i}"]

    print("Checkpoint loaded.")

    # 5. 단일 데이터 로드 & 추론
    graphs, text_id_dict = load_data(config.dataset, seed=config.seeds[0])
    for idx, data in enumerate(graphs[:10]):  # 예시: 앞 10개 데이터만
        result = inference(data, xs, model_list, prog_list, alpha_list, exit_list, device, T)
        print(result)