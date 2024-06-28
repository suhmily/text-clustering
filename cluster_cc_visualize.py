# %%
import pyarrow.parquet as pq
import json
from typing import List, Dict, Generator
from tqdm import tqdm
from datasets import Dataset
import matplotlib.pyplot as plt
from cycler import cycler
from src.text_clustering import ClusterClassifier
import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

# HDFS上Parquet文件的路径
hdfs_path = "viewfs://hadoop-lt-cluster/home/mmu_llm/dw/mmu_llm.db/customjtmath_2013_20/type=normal/part-04999-626445f5-ee23-4a80-b0bd-e35648f16988.c000.snappy.parquet"

def parse_json_content(json_str: str) -> str:
    try:
        json_data = json.loads(json_str)
        return json_data.get('content', '')
    except json.JSONDecodeError:
        return ''

def read_and_parse_parquet(file_path: str, max_chunks: int = None) -> Generator[Dict[str, str], None, None]:
    try:
        parquet_file = pq.ParquetFile(file_path)
        print(f"文件包含 {parquet_file.num_row_groups} 个行组")
        print(f"文件模式: {parquet_file.schema}")
        
        for i in tqdm(range(min(max_chunks or float('inf'), parquet_file.num_row_groups))):
            table = parquet_file.read_row_group(i)
            df = table.to_pandas()
            if 'text' in df.columns:
                for text in df['text']:
                    content = parse_json_content(text)
                    if content:
                        yield {'content': content}
            print(f"处理完第 {i+1} 个行组")
            if max_chunks and i + 1 >= max_chunks:
                print(f"已达到指定的最大块数 {max_chunks}，停止读取")
                break
    except Exception as e:
        print(f"读取文件时出错: {e}")

# 读取并解析数据
parsed_data = list(read_and_parse_parquet(hdfs_path, max_chunks=1))

if parsed_data:
    print(f"\n成功读取并解析数据")
    print(f"总共解析的数据条数: {len(parsed_data)}")
    print("前5条解析后的内容:")
    for item in parsed_data[:5]:
        print(item['content'][:100] + '...')  # 只打印每条内容的前100个字符

    # 将解析后的数据转换为Hugging Face Dataset格式
    dataset = Dataset.from_list(parsed_data)
else:
    print("无法读取或解析数据")


# %%
from datasets import config
# Configure HuggingFace datasets cache
config.HF_DATASETS_CACHE = "/mmu_nlp_hdd/suzhou03/data/model_zoo/hugging_face/datasets/cache"

# Create ./data directory (if it doesn't exist)
data_dir = "./data"
os.makedirs(data_dir, exist_ok=True)

# Assume dataset is already loaded and contains a 'content' column
texts = dataset["content"]

# Set custom color scheme
colors = [
    "#0F0A0A", "#FF6600", "#FFBE00", "#496767", "#87A19E",
    "#FF9200", "#0F3538", "#F8E08E", "#0F2021", "#FAFAF0"
]
plt.rcParams['axes.prop_cycle'] = cycler(color=colors)

# Create ClusterClassifier instance
cc = ClusterClassifier(embed_device="cuda")  # Use "cuda" if GPU is available, else use "cpu"

# Run clustering pipeline
print("Starting text clustering...")
embs, labels, summaries = cc.fit(texts)

# Display results
print("Clustering complete, generating visualization...")

# 注释掉 custom_show 函数的调用以进行分段调试
# def custom_show(embs, labels):
#     # Use PCA to reduce embeddings to 2D if they're not already
#     if embs.shape[1] > 2:
#         pca = PCA(n_components=2)
#         embs_2d = pca.fit_transform(embs)
#     else:
#         embs_2d = embs

#     fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    
#     unique_labels = np.unique(labels)
#     for i, label in enumerate(unique_labels):
#         mask = labels == label
#         ax.scatter(embs_2d[mask, 0], embs_2d[mask, 1], c=[colors[i % len(colors)]],
#                    label=f'Cluster {label}', s=0.75, alpha=0.8)
    
#     ax.legend()
#     ax.set_title("Text Clustering Visualization")
#     plt.show()

# Call the custom show function
# custom_show(embs, labels)

# Save results
save_path = os.path.join(data_dir, "cc_parquet_data")
cc.save(save_path)
print(f"Clustering model saved to {save_path}")

# Print and save cluster summaries
print("\nCluster Summaries:")
summary_path = os.path.join(data_dir, "cluster_summaries.txt")
with open(summary_path, 'w', encoding='utf-8') as f:
    for i, summary in enumerate(summaries):
        print(f"Cluster {i}: {summary}")
        f.write(f"Cluster {i}: {summary}\n")
print(f"Cluster summaries saved to {summary_path}")

# Example: Predict clusters for new texts
new_texts = ["This is a new math problem", "This is another text about history"]
cluster_labels, embeddings = cc.infer(new_texts, top_k=1)
print("\nCluster labels for new texts:")
for text, label in zip(new_texts, cluster_labels):
    print(f"Text: '{text}' -> Cluster: {label[0]}")

# Save clustering results to CSV file
results_df = pd.DataFrame({
    'text': texts,
    'cluster': labels
})
results_path = os.path.join(data_dir, "clustering_results.csv")
results_df.to_csv(results_path, index=False)
print(f"\nClustering results saved to {results_path}")

# Save visualization image
# plt.savefig(os.path.join(data_dir, "cluster_visualization.png"))
# print(f"Cluster visualization saved to {os.path.join(data_dir, 'cluster_visualization.png')}")

print("\nText clustering analysis complete. All output files have been saved to the ./data directory.")
