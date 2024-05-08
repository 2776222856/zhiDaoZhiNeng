# -*- coding: utf-8 -*-
import httpx
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import signal
import functools

from secret import api_key

JD_source_url = '../../dataBase/JD/processed_JD/output_output2_1-猎聘jd数据20000条.xlsx'
target_url = '../../dataBase/JD/vectorized_JD/vectorized_1-猎聘jd数据20000条.xlsx'

dimensions = ['管理与领导', '战略规划', '财务与预算', '市场营销与销售', '人力资源', '运营效率', '客户关系', '技术与创新',
              '质量控制', '物流与供应链', '教育与培训', '健康与安全', '研究与开发', '法律与合规', '公共关系与传媒']


# 需要挂新加坡的VPN才能使用这个GPT的api_key
client_gpt = OpenAI(
    api_key=api_key.gpt_api_key,
    http_client=httpx.Client(
        follow_redirects=True,
    ),
)
client_kimi = OpenAI(
    api_key=api_key.kimi_api_key,
    base_url="https://api.moonshot.cn/v1",
)
client_gpt2 = OpenAI(
    base_url='https://api.xty.app/v1',
    api_key=api_key.gpt2_api_key,
    http_client=httpx.Client(
        base_url="https://api.xty.app/v1",
        follow_redirects=True,
    ),
)

def read_JD_data(JD_url):
    # 指定需要读取的列
    columns_to_read = ['行业', '公司名称', '企业行业', '职责描述']
    JD_data = pd.read_excel(JD_url, usecols=columns_to_read)
    # 测试阶段取前20条数据
    JD_data = JD_data.head(20)
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
            completion = client_gpt2.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user",
                     "content": f'根据以下给出的企业和给出的维度信息，要求让职责描述对照维度进行向量化处理\n'
                                f'对用岗位的职责描述为：{JD_data['职责描述']}\n'
                                f'维度信息为：{dimensions}\n'
                                f'要求每个维度的评分在0到1之间，保留两位小数，描述与对应维度越匹配，对应维度得分越高\n'
                                f'要求输出格式为15个0到1之间的两位小数，每个数字之间用空格隔开区分.'
                                f'注意输出的只有数字和空格！不包含其他文本！'
                                f'请严格按照输出格式进行输出！'}
                ]
            )
            result = completion.choices[0].message.content
            # print(result)
            # 检查结果格式是否正确
            if check_result_format(result):
                return result
            else:
                print("结果格式不正确，重新执行处理过程。")
        except Exception as e:
            print(f'Error occurred: {e}')


# 保存处理过后的数据
def save_processed_data(processed_data, target_url):
    try:
        processed_data_df = pd.DataFrame(processed_data)  # 创建DataFrame
        processed_data_df.to_excel(target_url, index=False)
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
            result = gpt_process_JD_data(JD_data)
            decimals = result.split()
            for i, decimal in enumerate(decimals):
                row[dimensions[i]] = float(decimal)
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


if __name__ == '__main__':
    print('正在读取JD数据...')
    JD_data = read_JD_data(JD_source_url)
    print('获取完成')
    # print(JD_data['公司名称'])
    print('JD数据向量化...')
    processed_data = vectorize_JD_data(JD_data, target_url)
    print(f'JD数据已完成向量化!')
    print('正在保存处理结果...')
    save_processed_data(processed_data, target_url)
    print(f"已保存到文件'{target_url}'中")
