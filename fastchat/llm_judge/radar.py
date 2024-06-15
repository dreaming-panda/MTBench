import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


CATEGORIES = ["Writing", "Roleplay", "Reasoning", "Math", "Coding", "Extraction", "STEM", "Humanities"]


def get_model_df():
    cnt = 0
    q2result = []
    fin = open("/home/zhuominc/MGP2/FastChat/fastchat/llm_judge/data/mt_bench/model_judgment/gpt-4_single-origin.jsonl", "r")
    for line in fin:
        obj = json.loads(line)
        obj["category"] = CATEGORIES[(obj["question_id"]-81)//10]
        q2result.append(obj)
    df = pd.DataFrame(q2result)
    return df

def toggle(res_str):
    if res_str == "win":
        return "loss"
    elif res_str == "loss":
        return "win"
    return "tie"

def get_model_df_pair():
    fin = open("gpt-4_pair.jsonl", "r")
    cnt = 0
    q2result = []
    for line in fin:
        obj = json.loads(line)

        result = {}
        result["qid"] = str(obj["question_id"])
        result["turn"] = str(obj["turn"])
        if obj["g1_winner"] == "model_1" and obj["g2_winner"] == "model_1":
            result["result"] = "win"
        elif obj["g1_winner"] == "model_2" and obj["g2_winner"] == "model_2":
            result["result"] = "loss"
        else:
            result["result"] = "tie"
        result["category"] = CATEGORIES[(obj["question_id"]-81)//10]
        result["model"] = obj["model_1"]
        q2result.append(result)

    df = pd.DataFrame(q2result)

    return df

df = get_model_df()
#df_pair = get_model_df_pair()

all_models = df["model"].unique()
print(all_models)
scores_all = []
for model in all_models:
    for cat in CATEGORIES:
        # filter category/model, and score format error (<1% case)
        res = df[(df["category"]==cat) & (df["model"]==model) & (df["score"] >= 0)]
        score = res["score"].mean()

        # # pairwise result
        # res_pair = df_pair[(df_pair["category"]==cat) & (df_pair["model"]==model)]["result"].value_counts()
        # wincnt = res_pair["win"] if "win" in res_pair.index else 0
        # tiecnt = res_pair["tie"] if "tie" in res_pair.index else 0
        # winrate = wincnt/res_pair.sum()
        # winrate_adjusted = (wincnt + tiecnt)/res_pair.sum()
        # # print(winrate_adjusted)

        # scores_all.append({"model": model, "category": cat, "score": score, "winrate": winrate, "wtrate": winrate_adjusted})
        scores_all.append({"model": model, "category": cat, "score": score})

target_models = ["Meta-Llama-3-8B-Instruct-Normal", "Meta-Llama-3-8B-Instruct-Griffin"]

scores_target = [scores_all[i] for i in range(len(scores_all)) if scores_all[i]["model"] in target_models]

# sort by target_models
scores_target = sorted(scores_target, key=lambda x: target_models.index(x["model"]), reverse=True)

df_score = pd.DataFrame(scores_target)
df_score = df_score[df_score["model"].isin(target_models)]

rename_map = {"llama-13b": "LLaMA-13B",
              "alpaca-13b": "Alpaca-13B",
              "vicuna-33b-v1.3": "Vicuna-33B",
              "vicuna-13b-v1.3": "Vicuna-13B",
              "gpt-3.5-turbo": "GPT-3.5-turbo",
              "claude-v1": "Claude-v1",
              "gpt-4": "GPT-4"}

for k, v in rename_map.items():
    df_score.replace(k, v, inplace=True)

fig = px.line_polar(df_score, r = 'score', theta = 'category', line_close = True, category_orders = {"category": CATEGORIES},
                    color = 'model', markers=True, color_discrete_sequence=px.colors.qualitative.Pastel)

fig.update_layout(
    font=dict(
        size=18,
    ),
)
fig.write_image("original.png", width=800, height=600, scale=2)