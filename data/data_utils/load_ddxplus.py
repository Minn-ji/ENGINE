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
    entity_to_idx = {entity.lower().strip(): idx for idx, entity in enumerate(open(f"{data_path}entities.txt"))}

    graphs = []
    for i, file in enumerate(jsonl_files):
        with open(file, 'r') as f:
            for line in tqdm(f, total=len(file)):
                case_data = json.loads(line)
                symptom_entities = case_data['subgraph']['entities']
                disease_entities = [head for head, rel, tail in case_data['subgraph']['tuples']]
                subgraph_entities = symptom_entities + disease_entities
                subgraph_entities = list(set(subgraph_entities))
                local_entity_to_idx = {e: i for i, e in enumerate(subgraph_entities)}
                # 증상 + 질병 노드 엔티티 종합합
                num_nodes = len(subgraph_entities)

                # 노드 특징(x)을 텍스트 임베딩으로 생성
                # 엔티티 이름의 '_'를 공백으로 바꿔 모델이 더 잘 이해하도록
                formatted_entities = [name.replace('_', ' ') for name in subgraph_entities]

                # Sentence Transformer 모델을 사용해 텍스트를 임베딩 벡터로 변환
                with torch.no_grad(): # 그래디언트 계산 비활성화
                    embeddings = embedding_model.encode(formatted_entities, convert_to_tensor=True, device=device)
                
                # 서브그래프 내에 있는 증상 노드들에 대한 임베딩 생성
                x = embeddings.float()

                # 엣지 인덱스(edge_index) 생성 ---
                edge_list = []
                for head, rel, tail in case_data['subgraph']['tuples']:
                    if head in local_entity_to_idx and tail in local_entity_to_idx:
                        h_idx, t_idx =local_entity_to_idx[head], local_entity_to_idx[tail]
                        edge_list.append([h_idx, t_idx])
                        edge_list.append([t_idx, h_idx])  # 무방향 그래프

                if not edge_list:
                    edge_index = torch.empty((2, 0), dtype=torch.long)
                else:
                    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
                    edge_index = torch.unique(edge_index, dim=1)


                # 정답 레이블(y) 생성 ---
                if isinstance(case_data['answers'], list):
                    answer = case_data['answers'][0].lower().replace(" ", "_")
                else:
                    answer = case_data['answers'].lower().replace(" ", "_")
                answer_indices = []
                answer_labels = []

                if answer in local_entity_to_idx:
                    answer_indices.append(local_entity_to_idx[answer])
                for entity in subgraph_entities:
                    if entity == answer and entity.lower().strip() in local_entity_to_idx:
                        answer_labels.append(1)
                    else: 
                        answer_labels.append(-9999)
                idx_to_entity = {idx: ent for ent, idx in local_entity_to_idx.items()}
    
                y_result = [idx_to_entity[idx] for idx in answer_indices]
                print("answer_labels", answer_labels, y_result)

                y = torch.tensor(answer_labels, dtype=torch.long)

                # PyG Data 객체 생성
                data = Data(x=x, edge_index=edge_index, y=y)
                data.id = case_data['id']
                data.question = case_data['question']
                data.entities = subgraph_entities
                G = to_networkx(data, to_undirected=True)

                labels = {i: ent for i, ent in enumerate(data.entities)}

                # 노드 색 (평균값 같은 스칼라 필요)
                node_colors = []
                for idx in answer_labels:
                    if idx == 1:  
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
                os.makedirs('result_fig', exist_ok=True)
                plt.savefig(f"result_fig/graph_{case_data['id']}.png", dpi=300)

                graphs.append(data)
                break
            print(f"graph of id : {i}\n", graphs)
    return graphs[0]


    # # --- 7. 데이터 분할 (기존과 동일) ---
    # random.shuffle(graphs)
    # num_graphs = len(graphs)
    # train_end = int(num_graphs * 0.8)
    # val_end = int(num_graphs * 0.9)

    # train_graphs = graphs[:train_end]
    # val_graphs = graphs[train_end:val_end]
    # test_graphs = graphs[val_end:]

    # print(f"\nTotal graphs: {num_graphs}")
    # print(f"Train graphs: {len(train_graphs)}")
    # print(f"Validation graphs: {len(val_graphs)}")
    # print(f"Test graphs: {len(test_graphs)}")

    # return train_graphs, val_graphs, test_graphs

# --- 사용 예시 ---
if __name__ == '__main__':
    # GPU 사용 가능 여부 확인
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")

    # 데이터 파일이 있는 경로를 지정해주세요.
    DATASET_PATH = './gnn_data/' 
    
    try:
        graph = get_raw_text_ddxplus(data_path=DATASET_PATH, device=DEVICE)
        
        if graph:
            # 첫 번째 학습용 그래프 정보 출력
            print("\n--- Example Train Graph (with Text Embeddings) ---")
            first_graph = graph
            print(first_graph)
            # print(f"Graph ID: {first_graph.id}")
            # print(f"Number of nodes: {first_graph.num_nodes}")
            # # 특징 벡터의 차원이 노드 수가 아닌, 임베딩 모델의 차원(384)으로
            # print(f"Node features shape: {first_graph.x.shape}") 
            # print(f"Edge index shape: {first_graph.edge_index.shape}")
            # print(f"Label (answer node index): {first_graph.y}")
            # answer_node_name = [first_graph.entities[i] for i in first_graph.y]
            # print(f"Label (answer node name): {answer_node_name}")
        
    except FileNotFoundError:
        print(f"Error: Make sure '.jsonl' is in the '{DATASET_PATH}' directory.")
    except Exception as e:
        print(f"An error occurred: {e}")

# return data, text, num_classes
# data has train_mask, val_mask, test_mask
# 