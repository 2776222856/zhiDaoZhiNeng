# -*- coding: utf-8 -*-
import httpx
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from secret import api_key

source_url = "../../dataBase/New_JD_ONET_db.xlsx"
target_url = "../../dataBase/JD_ONET_checked.xlsx"


# 使用GPT-3.5检查指标信息
def gpt_check_result(jd_desc, onet_desc):
    while True:
        try:
            completion = api_key.client_gpt.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user",
                     "content": f''}
                ]
            )
            result = completion.choices[0].message.content
            return result
        except Exception as e:
            print("GPT执行出错")
            print(f'GPT Error occurred: {e}')


# 检查计算后的信息
def check_computed_data(computed_data):
    try:
        checked_data = []
        for _, row in tqdm(computed_data.iterrows(), total=len(computed_data), desc="检查匹配结果..."):
            for i in range(5):
                result = gpt_check_result(computed_data["职责描述"], computed_data[f'desc{i}'])
                row[f'score{i}'] = result
            checked_data.append(row)
        print('所有信息检查完成！')
        # 保存信息
        print('正在保存检查后的信息...')
        try:
            checked_data_df = pd.DataFrame(checked_data)
            checked_data_df.to_excel(target_url, index=False)
            print('信息保存成功！')
        except Exception as e2:
            print('保存已检查信息出错！')
            print(f'save checked error:{e2}')
    except Exception as e:
        print('检查信息出错')
        print(f'Check Error occurred: {e}')


def check_main():
    print('读取向量化计算结果...')
    computed_data = pd.read_excel(source_url)
    print('正在检查向量化匹配结果...')
    check_computed_data(computed_data)
    print('运行结束！')


if __name__ == "__main__":
    check_main()

