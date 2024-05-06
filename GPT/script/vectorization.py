# -*- coding: utf-8 -*-
import httpx
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import signal
import functools

from secret import api_key

dimensions = ['管理与领导', '战略规划', '财务与预算', '市场营销与销售', '人力资源', '运营效率', '客户关系', '技术与创新',
              '质量控制', '物流与供应链', '教育与培训', '健康与安全', '研究与开发', '法律与合规', '公共关系与传媒']
occupation_url = '../../dataBase/ONET/raw_db_28_2_excel/Occupation Data.xlsx'
target_url = '../../dataBase/ONET/process_db/Occupation Data Processed.xlsx'

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


def kimi_process_description(description):
    while True:
        completion = client_kimi.chat.completions.create(
            model="moonshot-v1-8k",
            messages=[
                {"role": "system",
                 "content": "你是 Kimi，由 Moonshot AI 提供的人工智能助手，你更擅长中文和英文的对话。你会为用户提供安全，有帮助，准确的回答。同时，你会拒绝一切涉及恐怖主义，种族歧视，黄色暴力等问题的回答。Moonshot AI 为专有名词，不可翻译成其他语言。"},
                {"role": "user",
                 "content": f'根据以下给出的职业描述信息和对应的维度信息，要求让职业描述对照维度进行向量化处理\n'
                            f'职业描述信息：{description}\n'
                            f'维度信息：{dimensions}\n'
                            f'要求每个维度的评分在0到1之间，保留两位小数，描述与对应维度越匹配，对应维度得分越高\n'
                            f'要求输出格式为：15个0到1之间的两位小数，每个数字之间用空格隔开区分'
                            f'注意输出的只有小数！不包含其他文本！'}
            ],
            temperature=0.3,
        )
        result = completion.choices[0].message.content
        return result
        # if check_result_format(result):
        #     return result
        # else:
        #     print("结果格式不正确，重新执行处理过程。")


def gpt_process_description(description):
    while True:
        try:
            # 使用GPT-3.5分析文本
            completion = client_gpt.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user",
                     "content": f'根据以下给出的职业描述信息和对应的维度信息，要求让职业描述对照维度进行向量化处理\n'
                                f'职业描述信息：{description}\n'
                                f'维度信息：{dimensions}\n'
                                f'要求每个维度的评分在0到1之间，保留两位小数，描述与对应维度越匹配，对应维度得分越高\n'
                                f'要求输出格式为15个0到1之间的两位小数，每个数字之间用空格隔开区分'
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


# 信号量机制实现数据自动保存
def save_data_and_exit(signal_num, frame, processed_data, target_url):
    # 保存已处理数据
    print("收到终止信号，正在保存已处理数据...")
    processed_data_df = pd.DataFrame(processed_data)
    processed_data_df.to_excel(target_url, index=False)
    print(f"已保存到文件'{target_url}'中")
    exit()


# 对ONET数据进行向量化操作
def vectorization_data(url, target_url):
    # 读取数据库文件
    data = pd.read_excel(url)
    processed_data = []
    # 读取上一次执行到的断点
    try:
        with open('checkpoint.txt', "r") as f:
            last_position = int(f.read())
    except FileNotFoundError:
        last_position = 0

    # 注册信号处理函数
    signal.signal(signal.SIGINT,
                  functools.partial(save_data_and_exit, processed_data=processed_data, target_url=target_url))
    signal.signal(signal.SIGTERM,
                  functools.partial(save_data_and_exit, processed_data=processed_data, target_url=target_url))

    # 处理每个元素
    try:
        for index, row in tqdm(data.iterrows(), total=len(data), desc="vectorization data"):
            if index < last_position:
                continue
            for column in data.columns:
                # 向量化处理
                if column == 'Description':
                    result = gpt_process_description(row[column])
                    # result = kimi_process_description(row[column])
                    # 将字符串结果拆分成小数
                    decimals = result.split()
                    # 保存每个小数到当前行的后续列中
                    for i, decimal in enumerate(decimals):
                        row[f'Decimal_{i + 1}'] = float(decimal)
                    processed_data.append(row)
                    # print('processed_data:')
                    # print(processed_data)
                    last_position += 1
        processed_data_df = pd.DataFrame(processed_data)  # 创建DataFrame
        processed_data_df.to_excel(target_url, index=False)
        print(f'文件{url}已完成向量化!')
        print(f"已保存到文件'{target_url}'中")
        return
    except Exception as e:
        print(f'Error occurred at index {index}: {e}')
        print(f'已成功处理{last_position}条数据')
        print('正在保存已处理数据...')
        processed_data_df = pd.DataFrame(processed_data)  # 创建DataFrame
        processed_data_df.to_excel(target_url, index=False)  # 保存到Excel
        print(f'已保存到文件{target_url}中')
        with open('checkpoint.txt', "w") as f:
            f.write(str(index + 1))  # 下次从下一个位置开始处理

    # for index, row in tqdm(data.iterrows(), total=len(data), desc="vectorization data"):
    #     if index < last_position:
    #         continue
    #     try:
    #         for column in data.columns:
    #             # 向量化处理
    #             if column == 'Description':
    #                 result = gpt_process_description(row[column])
    #                 # result = kimi_process_description(row[column])
    #                 print(result)
    #                 # return
    #                 # 将字符串结果拆分成小数
    #                 decimals = result.split()
    #                 # 保存每个小数到当前行的后续列中
    #                 for i, decimal in enumerate(decimals):
    #                     row[f'Decimal_{i + 1}'] = float(decimal)
    #                 processed_data.append(row)
    #                 print('processed_data:')
    #                 print(processed_data)
    #                 last_position += 1
    #
    #         processed_data_df = pd.DataFrame(processed_data)  # 创建DataFrame
    #         processed_data_df.to_excel(target_url, index=False)
    #     except Exception as e:
    #         print(f'Error occurred at index {index}: {e}')
    #         break  # 如果出现异常，跳出循环

        # 记录当前数据已执行
        # with open('checkpoint.txt', "w") as f:
        #     f.write(str(index + 1))  # 下次从下一个位置开始处理

    # # 将已处理的数据写入目标文件
    # processed_data_df.to_excel(target_url, index=False)


if __name__ == "__main__":
    vectorization_data(occupation_url, target_url)
    # print(api_key.gpt_api_key)
    # print(api_key.kimi_api_key)
    print('运行结束！')
