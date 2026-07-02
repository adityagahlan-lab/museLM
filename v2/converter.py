import pandas as pd
import ast
import re

print("Loading UltraChat...")
df = pd.read_csv("train_sft.csv")

count = 0

with open("dataset.txt", "a", encoding="utf-8") as f:

    for row in df["messages"]:

        try:
            # Add commas between dictionaries
            fixed = re.sub(r"}\s*{", "}, {", row)

            conversation = ast.literal_eval(fixed)

            for msg in conversation:
                role = msg["role"]
                text = msg["content"].replace("\n", " ").strip()

                if role == "user":
                    f.write(f"User: {text}\n")

                elif role == "assistant":
                    f.write(f"Assistant: {text}\n")

            f.write("<eos>\n\n")
            count += 1

        except Exception as e:
            print("Skipped:", e)
            continue

print(f"Done! Appended {count} conversations.")