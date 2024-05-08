# -*- coding: utf-8 -*-
import httpx
import pandas as pd
import numpy as np
from openai import OpenAI
from tqdm import tqdm
import signal
import functools

from secret import api_key

JD_source_url = '../../dataBase/JD/vectorized_JD/New vectorized_1-猎聘jd数据20000条.xlsx'
ONET_source_url = '../../dataBase/ONET/vectorized_ONET/New Occupation Data vectorized.xlsx'
target_url = '../../dataBase/New_JD_ONET_db.xlsx'


# 采用欧式距离进行计算
def compute_distances(JD_data, ONET_data, k=5):
    distances = []
    for _, jd_row in tqdm(JD_data.iterrows(), total=len(JD_data), desc="compute distance"):
        min_distances = []
        closest_onet_rows = []
        for _, onet_row in ONET_data.iterrows():
            # 提取向量值
            jd_vector = jd_row['vectorization'].values
            onet_vector = onet_row['vectorization'].values
            # 计算欧式距离
            distance = np.linalg.norm(jd_vector - onet_vector)
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
        final_rows_df.to_excel(target_url, index=False)
    except Exception as e:
        print('数据保存失败！')
        print(e)


def vector_main():
    print('正在读取JD数据...')
    JD_data = pd.read_excel(JD_source_url)
    print('正在读取ONET数据...')
    ONET_data = pd.read_excel(ONET_source_url)
    print('正在计算向量距离...')
    closest_distances = compute_distances(JD_data, ONET_data, k=5)
    print('计算完成!')
    print('正在保存数据...')
    save_result(closest_distances, target_url)
    print(f'数据已保存到{target_url}文件当中！')


if __name__ == '__main__':
    vector_main()


