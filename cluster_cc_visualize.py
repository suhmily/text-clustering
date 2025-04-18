# %%
import pyarrow.parquet as pq
import json
from typing import List, Dict, Generator
from tqdm import tqdm
from datasets import Dataset
import gc

# HDFS上Parquet文件的路径
hdfs_path = "viewfs://hadoop-lt-cluster/home/mmu_llm/dw/mmu_llm.db/customjtmath_2013_20/type=normal/part-04999-626445f5-ee23-4a80-b0bd-e35648f16988.c000.snappy.parquet"

def parse_json_content(json_str: str) -> str:
    try:
        json_data = json.loads(json_str)
        return json_data.get('content', '')
    except json.JSONDecodeError:
        return ''

def read_and_parse_parquet(file_path: str, max_chunks: int = None, max_length: int = 50) -> Generator[Dict[str, str], None, None]:
    try:
        parquet_file = pq.ParquetFile(file_path)
        print(f"文件包含 {parquet_file.num_row_groups} 个行组")
        print(f"文件模式: {parquet_file.schema}")
        
        for i in tqdm(range(min(max_chunks or float('inf'), parquet_file.num_row_groups))):
            table = parquet_file.read_row_group(i)
            df = table.to_pandas()
            if 'text' in df.columns:
                for text in df['text']:
                    content = parse_json_content(text)[:max_length]
                    if content:
                        yield content
            print(f"处理完第 {i+1} 个行组")
            if max_chunks and i + 1 >= max_chunks:
                print(f"已达到指定的最大块数 {max_chunks}，停止读取")
                break
    except Exception as e:
        print(f"读取文件时出错: {e}")

# 读取并解析数据
parsed_data = list(read_and_parse_parquet(hdfs_path, max_chunks=5))

if parsed_data:
    print(f"\n成功读取并解析数据")
    print(f"总共解析的数据条数: {len(parsed_data)}")
    print("前5条解析后的内容:")
    for item in parsed_data[:5]:
        print(item[:100] + '...')  # 只打印每条内容的前100个字符

    # 将解析后的数据转换为Hugging Face Dataset格式
    # dataset = Dataset.from_list(parsed_data)
    texts = parsed_data
    gc.collect()  # 强制进行垃圾回收

else:
    print("无法读取或解析数据")

# %%
import pandas as pd
from src.text_clustering import ClusterClassifier
from cycler import cycler
import matplotlib.pyplot as plt

# Ensure you have a pandas DataFrame named pandas_df
# Example:
# pandas_df = pd.DataFrame({'content': ["text1", "text2", "text3"]})

# Create an instance of ClusterClassifier
cc = ClusterClassifier(
    embed_device="cuda",
)

# Run the pipeline on the 'content' column

embs, labels, summaries = cc.fit(texts)
cc.optimize_parameters(target_noise_ratio=0.1, max_iterations=10)
# Plot k-distance graph for eps selection
cc.plot_k_distance(k=5)

# Customize color scheme (optional)
# default_cycler = (cycler(color=[
#     "#0F0A0A", "#FF6600", "#FFBE00", "#496767", "#87A19E",
#     "#FF9200", "#0F3538", "#F8E08E", "#0F2021", "#FAFAF0"
# ]))
# plt.rc('axes', prop_cycle=default_cycler)

# Visualize the results
cc.show(interactive=False)

# Save the classifier (optional)
cc.save("./content_clusters")


# If you want to classify new texts later:
# new_texts = ["Some new text", "Another new text"]
# cluster_labels, embeddings = cc.infer(new_texts, top_k=1)

# %%
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import numpy as np
from matplotlib.patches import Rectangle

def visualize_cluster_proportions(labels, summaries):
    # Count the number of items in each cluster
    cluster_counts = Counter(labels)
    
    # Sort the clusters by their labels, excluding -1 (which is typically used for noise)
    sorted_clusters = sorted([item for item in cluster_counts.items() if item[0] != -1])
    
    # Separate the labels and counts
    cluster_labels, counts = zip(*sorted_clusters)
    
    # Calculate the proportion of noise points (label -1)
    noise_count = cluster_counts.get(-1, 0)
    total_count = sum(counts) + noise_count
    noise_proportion = noise_count / total_count if total_count > 0 else 0
    
    # Calculate percentages
    percentages = [count / sum(counts) * 100 for count in counts]
    
    # Create a color palette
    colors = sns.color_palette("husl", len(cluster_labels))
    
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), gridspec_kw={'width_ratios': [1, 1]})
    
    # Create a pie chart on the left subplot
    wedges, texts, autotexts = ax1.pie(counts, autopct='%1.1f%%', startangle=90, colors=colors,
                                       textprops=dict(color="w", weight="bold", size=10))
    
    # Add title to the pie chart
    ax1.set_title('Proportion of Data Points in Each Cluster', fontsize=16)
    
    # Add a note about noise proportion
    ax1.annotate(f'Noise: {noise_proportion:.1%}', xy=(0.95, 0.05), xycoords='axes fraction',
                 horizontalalignment='right', verticalalignment='bottom', fontsize=10)
    
    # Ensure the pie chart is circular
    ax1.axis('equal')
    
    # Create colored text with percentages for each cluster on the right subplot
    ax2.set_ylim(0, 1)
    ax2.set_xlim(0, 1)
    ax2.axis('off')  # Turn off axes for the text subplot
    
    for i, (label, summary, color, percentage) in enumerate(zip(cluster_labels, summaries.values(), colors, percentages)):
        y_position = 1 - (i + 1) / (len(cluster_labels) + 1)
        rect = Rectangle((0, y_position), 0.03, 0.03, facecolor=color, edgecolor='none')
        ax2.add_patch(rect)
        ax2.text(0.05, y_position, f'Cluster {label} ({percentage:.1f}%): {summary}', 
                 va='center', ha='left', wrap=True, fontsize=12)
    
    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()
    # labels = cc.cluster_labels
# summaries = cc.cluster_summaries
visualize_cluster_proportions(labels, summaries)

# %%
summaries

# %%

noise_ratio = cc.calculate_noise_ratio()
noise_ratio


