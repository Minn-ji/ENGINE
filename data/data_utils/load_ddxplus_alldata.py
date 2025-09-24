import os
import glob
import json
import torch
import random
from tqdm import tqdm
import numpy as np
from torch_geometric.data import Data
from sentence_transformers import SentenceTransformer

import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx



# data = Data(x=x, edge_index=edge_index)

# # PyTorch Geometric의 그래프를 NetworkX로 변환
# G = to_networkx(data, to_undirected=True)

# # 노드의 특징 (x 값을 노드의 색으로 표현)
# node_colors = [data.x[i].item() for i in range(data.num_nodes)]

# # 그래프 시각화
# plt.figure(figsize=(8, 6))
# nx.draw(G, with_labels=True, node_color=node_colors, cmap=plt.cm.Blues, node_size=500, font_size=16)
#os.makedirs('result_fig', exist_ok=True)
# plt.savefig(f"result_fig/graph_{case_data['id']}.png", dpi=300)

def get_raw_text_ddxplus(data_path='./gnn_data/', SEED=0, device='cpu'):
    """
    ddxplus 데이터셋을 로드하여 각 케이스를 PyTorch Geometric의 Data 객체 리스트로 반환합니다.
    노드 특징은 Sentence Transformer를 이용한 텍스트 임베딩을 사용합니다.
    데이터는 60/20/20 비율로 train/validation/test 세트로 분할됩니다.

    Args:
        data_path (str): 데이터 파일이 위치한 경로.
        SEED (int): 데이터 분할을 위한 랜덤 시드.
        device (str): 임베딩 모델을 실행할 장치 ('cuda' 또는 'cpu').

    Returns:
        tuple: (train_graphs, val_graphs, test_graphs)
               각각 PyG Data 객체의 리스트
    """
    # --- 1. 기본 설정 및 임베딩 모델 로드 ---
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # 사전 학습된 언어 모델 로드. 처음 실행 시 모델을 다운로드
    print("Loading Sentence Transformer model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    print("Model loaded.")

    jsonl_files = glob.glob(os.path.join(data_path, "*.jsonl"))
    jsonl_files = [jsonl_files[0]]
    # 엔티티 매핑 (전체 entity space)
    with open(f"{data_path}entities.txt") as f:
        global_entities = [line.strip().lower() for line in f if line.strip()]
    entity_to_idx = {entity.lower().strip(): idx for idx, entity in enumerate(global_entities)}
    all_edges = []
    all_labels = []
    for i, file in enumerate(jsonl_files):
        with open(file, 'r') as f:
            for line in tqdm(f, total=len(file)):
                case_data = json.loads(line)
                # --- 2. 엔티티(노드) 및 인덱스 매핑 생성 ---
                subgraph_entities = case_data['subgraph']['entities']
                local_map = {e: entity_to_idx[e] for e in subgraph_entities if e in entity_to_idx}

                for head, rel, tail in case_data['subgraph']['tuples']:
                    if head in local_map and tail in local_map:
                        h, t = local_map[head], local_map[tail]
                        all_edges.append([h, t])
                        all_edges.append([t, h])

                for ans in case_data['answers']:
                    if ans.lower().replace(' ', '_') in local_map:
                        all_labels.append(entity_to_idx[ans.lower().replace(' ', '_')]) # 질병만 들어감
  
    edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
    formatted_entities = [name.replace('_', ' ') for name in global_entities]
    with torch.no_grad():
        # 3. 노드 인덱스 생성
        x = embedding_model.encode(global_entities, convert_to_tensor=True, device=device).float()
    y = torch.tensor(all_labels, dtype=torch.long) 
    idx_to_entity = {idx: ent for ent, idx in entity_to_idx.items()}
    print("label의 개수 : ", len(set(all_labels)),"\n", [idx_to_entity[idx] for idx in list(set(all_labels))])


    # --- 6. PyG Data 객체 생성 (기존과 동일) ---
    data = Data(x=x, edge_index=edge_index, y=y)
    data.id = case_data['id']
    data.question = case_data['question']
    data.entities = global_entities
    data.num_target_nodes = len(all_labels)
    data.num_nodes = len(global_entities)

    ####################### 시각화 용 ########################
    G = to_networkx(data, to_undirected=True)
    labels = {i: ent for i, ent in enumerate(data.entities)}

    node_colors = []
    for idx in range(len(data.entities)):
        if idx in all_labels:  
            node_colors.append("blue")       # 정답 노드 
        else:
            node_colors.append("lightgray") # 나머지 → 회색
    pos=nx.spring_layout(G, k=0.3)
    plt.figure(figsize=(10, 8))
    nx.draw(
        G,
        pos,
        labels=labels,                   # 엔티티 이름으로 라벨 표시
        node_color=node_colors,
        cmap=plt.cm.Blues,
        node_size=1000,
        font_size=10,
        font_color="black"
    )
    plt.savefig("graph_all.png", dpi=300)
    #####################################################

    # 라벨이 존재하는 노드만 필터링 (질병 노드만)
    labeled_nodes = torch.nonzero(data.y != -1, as_tuple=False).squeeze() 
    labeled_nodes = labeled_nodes.numpy()
    np.random.shuffle(labeled_nodes)

    node_id = np.arange(data.num_nodes)
    np.random.shuffle(node_id)

    train_end = int(len(labeled_nodes) * 0.6)
    val_end   = int(len(labeled_nodes) * 0.8)
    data.train_id = labeled_nodes[:train_end]
    data.val_id   = labeled_nodes[train_end:val_end]
    data.test_id  = labeled_nodes[val_end:]
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.val_mask   = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask  = torch.zeros(data.num_nodes, dtype=torch.bool)

    data.train_mask[data.train_id] = True
    data.val_mask[data.val_id]     = True
    data.test_mask[data.test_id]   = True


    # return train_graphs, val_graphs, test_graphs
    return data, data.entities, data.num_nodes
# Graph data, text, num_classes
# --- 사용 예시 ---
if __name__ == '__main__':
    # GPU 사용 가능 여부 확인
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")

    # 데이터 파일이 있는 경로를 지정해주세요.
    DATASET_PATH = './gnn_data/' 
    
    try:
        data = get_raw_text_ddxplus(data_path=DATASET_PATH, device=DEVICE)
        
        if data:
            # 첫 번째 학습용 그래프 정보 출력
            print("\n--- Example Train Graph (with Text Embeddings) ---")
            graph = data[0]
            print(graph)
            print(f"Graph ID: {graph.id}")
            print(f"Number of nodes: {graph.num_nodes}")
            # 특징 벡터의 차원이 노드 수가 아닌, 임베딩 모델의 차원(384)으로
            print(f"Node features shape: {graph.x.shape}") 
            print(f"Edge index shape: {graph.edge_index.shape}")
            print(f"Label (answer node index): {graph.y}")
            answer_node_name = [graph.entities[i] for i in graph.y]
            print(f"Label (answer node name): {answer_node_name}")
        
    except FileNotFoundError:
        print(f"Error: Make sure '.jsonl' is in the '{DATASET_PATH}' directory.")
    except Exception as e:
        print(f"An error occurred: {e}")

# return data, text, num_classes
# data has train_mask, val_mask, test_mask
# 