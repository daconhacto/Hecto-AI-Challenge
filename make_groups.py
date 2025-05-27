import os
import json
from collections import defaultdict

def find_class_groups_from_jsons(folder_path, start_epoch = 10):
    # 1. 클래스 간 연결 정보 추출
    class_graph = defaultdict(set)

    for file_name in os.listdir(folder_path):
        if int(file_name.split('_')[3]) < start_epoch:
            continue

        if file_name.endswith(".json"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

                for true_class, examples in data.items():
                    for ex in examples:
                        predicted_class = ex['model_answer']
                        # 양방향 연결로 그래프 구성
                        class_graph[true_class].add(predicted_class)
                        class_graph[predicted_class].add(true_class)

    # 2. 그래프 기반 그룹 찾기 (연결된 component 별로 묶기)
    visited = set()
    groups = []

    def dfs(node, group):
        visited.add(node)
        group.append(node)
        for neighbor in class_graph[node]:
            if neighbor not in visited:
                dfs(neighbor, group)

    for cls in class_graph:
        if cls not in visited:
            group = []
            dfs(cls, group)
            groups.append(sorted(group))

    return groups


def main():
    work_dir = '/project/ahnailab/jys0207/CP/tjrgus5/hecto/work_directories/eva_mosaic_mixup_cutmix'
    start_epoch = 15

    wrong_examples = os.path.join(work_dir, 'wrong_examples')
    groups = find_class_groups_from_jsons(wrong_examples, start_epoch)
    with open(os.path.join(work_dir, 'groups.json'), 'w') as f:
        json.dump(groups, f)

if __name__ == "__main__":
    main()