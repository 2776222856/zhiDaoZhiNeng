# -*- coding: utf-8 -*-
import httpx
import pandas as pd
import numpy as np
import ast
from openai import OpenAI
from tqdm import tqdm
import signal
import functools
import pdb

from secret import api_key

JD_source_url = '../../dataBase/JD/vectorized_JD/New vectorized_1-猎聘jd数据20000条.csv'
ONET_source_url = '../../dataBase/ONET/vectorized_ONET/New Occupation Data vectorized.csv'
target_url = '../../dataBase/New_JD_ONET_db_Cosine.csv'


# 采用欧式距离进行计算
def compute_distances(JD_data, ONET_data, k=5):
    distances = []
    for _, jd_row in tqdm(JD_data.iterrows(), total=len(JD_data), desc="compute distance"):
        min_distances = []
        closest_onet_rows = []
        for _, onet_row in ONET_data.iterrows():
            # 提取向量值并转换为NumPy数组
            jd_vector = np.array(ast.literal_eval(jd_row['vectorization']))
            onet_vector = np.array(ast.literal_eval(onet_row['vectorization']))
            # print("JD Vector Length:", jd_vector.shape[0])
            # print("ONET Vector Length:", onet_vector.shape[0])
            # 计算欧式距离
            distance = np.linalg.norm(jd_vector - onet_vector)
            # print("distance:", distance)
            # 更新最小距离列表和最近数据列表
            if len(min_distances) < k:
                min_distances.append(distance)
                closest_onet_rows.append(onet_row)
            else:
                max_distance_index = np.argmax(min_distances)
                if distance < min_distances[max_distance_index]:
                    min_distances[max_distance_index] = distance
                    closest_onet_rows[max_distance_index] = onet_row
        # 对 min_distances 列表进行排序
        min_distances_sorted_indices = np.argsort(min_distances)
        min_distances_sorted = [min_distances[i] for i in min_distances_sorted_indices]
        closest_onet_rows_sorted = [closest_onet_rows[i] for i in min_distances_sorted_indices]
        # jd_row仅保留前四列数据
        jd_row_selected = jd_row[:4]
        distances.append((jd_row_selected, closest_onet_rows_sorted, min_distances_sorted))
    return distances


# 采用Jaccard相似度进行计算
def compute_Jaccard_Similarity(JD_data, ONET_data, k=5):
    similarity = []
    for _, jd_row in tqdm(JD_data.iterrows(), total=len(JD_data), desc="compute distance"):
        max_similarity = []
        closest_onet_rows = []
        for _, onet_row in ONET_data.iterrows():
            # 提取向量值并转换为集合
            jd_vector = set(ast.literal_eval(jd_row['vectorization']))
            onet_vector = set(ast.literal_eval(onet_row['vectorization']))
            # 计算Jaccard相似度
            intersection = len(jd_vector.intersection(onet_vector))
            union = len(jd_vector.union(onet_vector))
            jaccard_similarity = intersection / union
            # 更新最小相似度列表和最近数据列表
            if len(max_similarity) < k:
                max_similarity.append(jaccard_similarity)
                closest_onet_rows.append(onet_row)
            else:
                min_similarity_index = np.argmax(max_similarity)
                if jaccard_similarity > max_similarity[min_similarity_index]:
                    max_similarity[min_similarity_index] = jaccard_similarity
                    closest_onet_rows[min_similarity_index] = onet_row
        # 对 min_distances 列表进行排序
        max_similarity_sorted_indices = np.argsort(max_similarity)[::-1]  # 对相似度从高到低排序
        max_similarity_sorted = [max_similarity[i] for i in max_similarity_sorted_indices]
        closest_onet_rows_sorted = [closest_onet_rows[i] for i in max_similarity_sorted_indices]
        # jd_row仅保留前四列数据
        jd_row_selected = jd_row[:4]
        similarity.append((jd_row_selected, closest_onet_rows_sorted[:k], max_similarity_sorted[:k]))
    return similarity


# 采用余弦距离进行计算
def compute_Cosine_Similarity(JD_data, ONET_data, k=5):
    similarity = []
    for _, jd_row in tqdm(JD_data.iterrows(), total=len(JD_data), desc="compute distance"):
        max_similarity = []
        closest_onet_rows = []
        print('-----------------------------余弦相似度--------------------------------')
        for _, onet_row in ONET_data.iterrows():
            # 提取向量值并转换为numpy数组
            jd_vector = np.array(ast.literal_eval(jd_row['vectorization']))
            onet_vector = np.array(ast.literal_eval(onet_row['vectorization']))
            # 计算余弦相似度
            cosine_similarity = np.dot(jd_vector, onet_vector) / (np.linalg.norm(jd_vector) * np.linalg.norm(onet_vector))
            # 更新最大相似度列表和最近数据列表
            print(cosine_similarity)
            if len(max_similarity) < k:
                max_similarity.append(cosine_similarity)
                closest_onet_rows.append(onet_row)
            else:
                min_similarity_index = np.argmin(max_similarity)
                # min_similarity_index = np.argmax(max_similarity)
                if cosine_similarity > max_similarity[min_similarity_index]:
                    max_similarity[min_similarity_index] = cosine_similarity
                    closest_onet_rows[min_similarity_index] = onet_row
        # 对 max_similarity 列表进行排序
        pdb.set_trace()
        max_similarity_sorted_indices = np.argsort(max_similarity)[::-1]  # 对相似度从高到低排序
        # max_similarity_sorted_indices = np.argsort(max_similarity)  # 对相似度从低到高排序

        max_similarity_sorted = [max_similarity[i] for i in max_similarity_sorted_indices]
        closest_onet_rows_sorted = [closest_onet_rows[i] for i in max_similarity_sorted_indices]
        # jd_row仅保留前四列数据
        jd_row_selected = jd_row[:4]
        similarity.append((jd_row_selected, closest_onet_rows_sorted[:k], max_similarity_sorted[:k]))
        print('-----------------------------匹配结果--------------------------------')
        print(similarity)
        pdb.set_trace()
    return similarity


# 保存处理完成后的结果
def save_result(closest_distances, target_url):
    try:
        final_rows = []
        # 对结果进行初步处理
        for jd_row, closest_onet_rows, min_distances in closest_distances:
            # print("JD 数据:", jd_row)
            # print("最近的 5 条 ONET 数据:")
            for i in range(len(closest_onet_rows)):
                jd_row[f'ONET{i}'] = closest_onet_rows[i]['O*NET-SOC Code']
                jd_row[f'desc{i}'] = closest_onet_rows[i]['Description']
                jd_row[f'dist{i}'] = min_distances[i]
                jd_row[f'score{i}'] = '未评分'
                # print("    数据:", closest_onet_rows[i])
                # print("    距离:", min_distances[i])
            final_rows.append(jd_row)
            # print()
        # 保存到target_url中
        final_rows_df = pd.DataFrame(final_rows)
        final_rows_df.to_csv(target_url, index=False)
    except Exception as e:
        print('数据保存失败！')
        print(e)


def vector_main():
    print('正在读取JD数据...')
    JD_data = pd.read_csv(JD_source_url)
    print('正在读取ONET数据...')
    ONET_data = pd.read_csv(ONET_source_url)
    print('正在计算向量距离...')
    # closest_distances = compute_distances(JD_data, ONET_data, k=5)
    # print('正在计算Jaccard相似度...')
    # closest_distances = compute_Jaccard_Similarity(JD_data, ONET_data, k=5)
    closest_distances = compute_Cosine_Similarity(JD_data, ONET_data, k=5)
    print('计算完成!')
    print('正在保存数据...')
    save_result(closest_distances, target_url)
    print(f'数据已保存到{target_url}文件当中！')


if __name__ == '__main__':
    vector_main()


