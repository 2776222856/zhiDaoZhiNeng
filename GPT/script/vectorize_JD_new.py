# -*- coding: utf-8 -*-
import httpx
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import signal
import functools

from secret import api_key

JD_source_url = '../../dataBase/JD/processed_JD/output_output2_1-猎聘jd数据20000条.xlsx'
target_url = '../../dataBase/JD/vectorized_JD/New vectorized_1-猎聘jd数据20000条.xlsx'
target_csv_url = '../../dataBase/JD/vectorized_JD/New vectorized_1-猎聘jd数据20000条.csv'

dimensions = ['管理与领导', '战略规划', '财务与预算', '市场营销与销售', '人力资源', '运营效率', '客户关系', '技术与创新',
              '质量控制', '物流与供应链', '教育与培训', '健康与安全', '研究与开发', '法律与合规', '公共关系与传媒']


def read_JD_data(JD_url):
    # 指定需要读取的列
    columns_to_read = ['行业', '公司名称', '企业行业', '职责描述']
    JD_data = pd.read_excel(JD_url, usecols=columns_to_read)
    # 测试阶段取前2000条数据
    JD_data = JD_data.head(2000)
    # print(JD_data)
    return JD_data


# 检查处理结果的格式是否正确
def check_result_format(result):
    # 将结果字符串按空格分割成列表
    decimals = result.split()
    # 检查列表长度是否为15
    if len(decimals) != 15:
        return False
    # 检查每个元素是否为数字
    for decimal in decimals:
        try:
            float(decimal)
        except ValueError:
            return False
    return True


def gpt_process_JD_data(JD_data):
    while True:
        try:
            # 使用GPT-3.5分析文本
            response = api_key.client_gpt.embeddings.create(
                input=f'{JD_data["职责描述"]}',
                model='text-embedding-3-small',
            )
            result = response.data[0].embedding
            # print(result)
            return result
        except Exception as e:
            print('gpt 处理出现错误')
            print(f'Error occurred: {e}')


# 保存处理过后的数据
def save_processed_data(processed_data, target_url):
    try:
        processed_data_df = pd.DataFrame(processed_data)  # 创建DataFrame
        processed_data_df.to_csv(target_url, index=False)
        # processed_data_df.to_excel(target_url, index=False)
        return
    except Exception as e:
        print("文件保存失败！")
        print(e)
        try:
            # 手动创建error_processed_data.csv文件，将processed_data写入其中
            error_target_url = 'error_processed_data.csv'
            error_processed_data_df = pd.DataFrame(processed_data)
            error_processed_data_df.to_csv(error_target_url, index=False)
            print(f"内容已备份到'{error_target_url}'中")
        except Exception as e2:
            print("文件备份失败！")
            print(e2)


def vectorize_JD_data(JD_data, target_url):
    processed_data = []
    # 读取上一次执行到的断点
    try:
        with open('checkpoint2.txt', "r") as f:
            last_position = int(f.read())
    except FileNotFoundError:
        last_position = 0
    # 遍历数据并向量化
    try:
        for index, row in tqdm(JD_data.iterrows(), total=len(JD_data), desc="vectorization JD data"):
            if index < last_position:
                continue
            row['职责描述'] = row['职责描述'].replace("\n", " ")
            result = gpt_process_JD_data(JD_data)
            # decimals = result.split()
            # for i, decimal in enumerate(decimals):
            #     row[dimensions[i]] = float(decimal)
            row['vectorization'] = result
            processed_data.append(row)
            last_position += 1
        return processed_data
    except Exception as e:
        print(f'Error occurred at index {index}: {e}')
        print(f'已成功处理{last_position}条数据')
        print('正在保存已处理数据...')
        save_processed_data(processed_data, target_url)
        with open('checkpoint.txt', "w") as f:
            f.write(str(index + 1))  # 下次从下一个位置开始处理


def JD_main():
    print('正在读取JD数据...')
    JD_data = read_JD_data(JD_source_url)
    print('获取完成')
    # print(JD_data['公司名称'])
    print('JD数据向量化...')
    processed_data = vectorize_JD_data(JD_data, target_csv_url)
    print(f'JD数据已完成向量化!')
    print('正在保存处理结果...')
    save_processed_data(processed_data, target_csv_url)
    print(f"已保存到文件'{target_csv_url}'中")


if __name__ == '__main__':
    JD_main()
