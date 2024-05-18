import pandas as pd
import numpy as np
import ast
from openai import OpenAI
from secret import api_key

client = OpenAI(
    api_key=api_key.gpt_api_key
)


def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


text = ("Determine and formulate policies and provide overall direction of companies or private and public sector "
        "organizations within guidelines set up by a board of directors or similar governing body. Plan, direct, "
        "or coordinate operational activities at the highest level of management with the help of subordinate "
        "executives and staff managers.")

embedding_result = get_embedding(text)
print('向量化结果：', embedding_result)

text_df = pd.DataFrame(embedding_result)
text_df.to_excel('test.xlsx')
text_df.to_csv('test.csv')

text_vector = np.array(ast.literal_eval(str(embedding_result)))
print('向量长度：', text_vector.shape[0])
