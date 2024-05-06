from pathlib import Path
from openai import OpenAI

client = OpenAI(
    api_key="sk-bdfg17mlHdTlKzqGKkcCJkS3mlXYZA1SSjR7fb3uocHgBopi",
    base_url="https://api.moonshot.cn/v1",
)

files_path = [
    # '../../dataBase/JD/1-猎聘jd数据20000条.xlsx',
    # '../../dataBase/JD/2-猎聘jd数据10000条.xlsx',
    # '../../dataBase/JD/3-猎聘jd数据10000条.xlsx',
    # '../../dataBase/JD/4-猎聘jd数据7618条.xlsx',
    # '../../dataBase/JD/前程无忧jd数据19125条.xlsx',
    # '../../dataBase/jd_info_processed/猎聘jd数据20000条预处理.xlsx',
    # '../../dataBase/jd_info_processed/猎聘jd数据5000条预处理.xlsx',
    '../../dataBase/jd_info_processed/猎聘jd数据2000条预处理.xlsx',
]

target_path = '../result'

for file_path in files_path:
    print('正在处理'+file_path+'\n')
    # 上传文件
    file_object = client.files.create(file=Path(file_path), purpose="file-extract")

    # 获取结果
    file_content = client.files.retrieve_content(file_id=file_object.id)
    # 注意，之前 retrieve_content api 在最新版本标记了 warning, 可以用下面这行代替
    # 如果是旧版本，可以用 retrieve_content
    # file_content = client.files.content(file_id=file_object.id).text

    # 把它放进请求中
    messages = [
        {
            "role": "system",
            "content": "你是 Kimi，由 Moonshot AI 提供的人工智能助手，你更擅长中文和英文的对话。你会为用户提供安全，有帮助，准确的回答。同时，你会拒绝一切涉及恐怖主义，种族歧视，黄色暴力等问题的回答。Moonshot AI 为专有名词，不可翻译成其他语言。",
        },
        {
            "role": "system",
            "content": file_content,
        },
        {"role": "user", "content": "根据如下文件的新增数据，尤其结合其中的职业介绍和职业描述相关数据，建立职业树，并且对现有分类进行改善和补充"},
    ]

    # 然后调用 chat-completion, 获取 kimi 的回答
    completion = client.chat.completions.create(
        model="moonshot-v1-32k",
        messages=messages,
        temperature=0.3,
    )
    print('文件' + file_path + '处理完成\n')
    print(completion.choices[0].message+'\n')

    # 存储输出结果
