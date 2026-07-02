import pandas as pd
import ast

df = pd.read_csv("train_sft.csv")

try:
    ast.literal_eval(df["messages"][0])
except Exception as e:
    print(e)