# import matplotlib.pyplot as plt
# from sklearn.tree import export_text
# from sklearn import tree
from joblib import load
from sklearn.tree import export_graphviz
import graphviz
import re

name = "viper_ToyPong-v0_pruned_1e-06_11120_29"
file = "log/" + name + ".joblib"

KEY_TO_LABEL = {
    0: 'paddle_x',
    1: 'ball_pos_x',
    2: 'ball_pos_y',
    3: 'ball_vel_x',
    4: 'ball_vel_y'
}

# Load cây từ file tree.viper
tree = load(file)

print(type(tree))
print(tree)

print(tree.classes_)

def modify_dot(dot_data):
    lines = dot_data.split("\n")
    new_lines = []
    
    # Tạo danh sách các node có con (không phải node lá)
    parent_nodes = set()
    for line in lines:
        match = re.search(r'(\d+) -> (\d+)', line)
        if match:
            parent_nodes.add(match.group(1))  # Lưu node cha vào danh sách

    for line in lines:
        if "class =" in line:  # Chỉ xử lý các dòng có class
            match = re.search(r'(\d+) \[', line)  # Lấy ID của node này
            if match and match.group(1) in parent_nodes:
                line = re.sub(r"class = \w+", "", line)  # Xóa class nếu là node phân nhánh

        new_lines.append(line)
    
    return "\n".join(new_lines)


dot_data = export_graphviz(
    tree,
    out_file=None,
    filled=True,
    rounded=True,
    special_characters=True,
    feature_names=[KEY_TO_LABEL[i] for i in range(len(KEY_TO_LABEL))],  # Thay số bằng tên feature
    class_names = ["left", "right", "stay"]  
)

# Hiển thị cây
dot_data = modify_dot(dot_data)
graph = graphviz.Source(dot_data)
graph.render("decision_tree_" + name)  # Xuất file PNG
# graph.view() 

# from tqdm import tqdm
# import time

# # Ví dụ kiểm tra thanh tiến trình
# for i in tqdm(range(10), desc="Đang tiến hành"):
#     time.sleep(0.1)