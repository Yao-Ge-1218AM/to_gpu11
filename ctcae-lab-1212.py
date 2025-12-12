import pandas as pd
import time
import json
from openai import AzureOpenAI

# ✅ 初始化 AzureOpenAI 客户端
client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint="https://bionlp-ge.openai.azure.com/",
    api_key=""   # ←← 换成你的 key
)

# ✅ 读取 lab 文件 (tsv)
df = pd.read_csv("/data/gey2/ctcae-data/Lab266632-1_04.csv")

mrn_col = "hash"

# ✅ Prompt 模板
base_prompt = f"""You are a clinical research assistant helping to extract adverse events (AEs) from lab data. Pay special attention to the rise and fall of some values, such as Lymphocyte. For neutrophils, Lymphopenias and Leukopenia, always output physiologic descriptions (e.g., “Neutrophil count decreased”, “Lymphocyte count decreased”, “White blood cell decreased") instead of CTCAE terms like “Neutropenia” or “Lymphopenia”. For all other lab-based AEs (e.g., Anemia, Eosinophilia), use standard CTCAE terms.

Lab summary:
<text>
{{text}}
</text>

For each AE, extract the following fields **in JSON array format** (one object per AE):

- MRN (from the note)
- Onset Date: If a specific start date is mentioned, extract it directly; otherwise, use the clinic note date as the start date or estimate an onset date according to the notes. "Onset Date" MUST NEVER be "Unknown" or "unknown". 
- Date Resolved: If a specific end date or resolution (“…has resolved”) is mentioned, extract it; for events like “weight loss → gain weight,” use the clinic note date as the end date. If the AE is described as ongoing, set end date to “ongoing.” If not mentioned, set end date to “unknown.”
- AE term (mapped to CTCAE terminology)
- Grade (must be 1 to 5). For lab-based AEs, estimate the grade based on the *severity of the lab abnormality and the need for clinical intervention*, following CTCAE-style reasoning:
    - Grade 1: Mild, asymptomatic or mild lab abnormality, close to the reference limit, usually not requiring intervention.
    - Grade 2: Moderate, clearly abnormal lab value or persistent trend that requires minimal/standard intervention (e.g., medication adjustment, temporary dose hold, closer monitoring) but not hospitalization.
    - Grade 3: Severe lab abnormality or rapid worsening requiring significant intervention (e.g., transfusion, IV medications, high-dose steroids, or hospitalization).
    - Grade 4: Life-threatening lab abnormality with critical risk (e.g., extremely low/high values requiring urgent intervention or ICU-level care).
    - Grade 5: Death related to the lab abnormality.
  If grade is not explicitly stated, infer the most appropriate grade based on the extent of lab abnormality, trend over time, and any described clinical actions.
- Attribution to Disease? One of [Unrelated, Unlikely, Possible, Probable, and Definite]
- Immune-related AE? (Yes/No): Mark “Yes” if the AE is immune-related (irAE) based on the following definition.
Definition of immune-related adverse events (irAEs): irAEs are adverse events relevant to immunotherapy, such as colitis, thyroiditis, hypophysitis, adrenalitis, myositis, myocarditis, encephalitis, pneumonitis, hepatitis, immunotherapy-induced diabetes mellitus, vitiligo, and similar conditions. If the AE is immune-mediated or commonly recognized as an irAE, mark “Yes”; otherwise, mark “No”.
- serious AE? (Yes/No) Mark “Yes” if the AE is considered serious (e.g., life-threatening, hospitalization, or significant disability); otherwise, “No.”

If no adverse events are present, return an empty JSON array: []
Do NOT include any explanation. Only return the JSON array.

Return a JSON array. Each AE MUST be a JSON object with EXACTLY the following keys
(using the same spelling and capitalization):

[
  {{
    "MRN": "...",
    "Onset Date": "...",
    "Date Resolved": "...",
    "AE Term": "...",
    "Grade": "...",
    "Attribution to Disease": "...",
    "Immune-related AE": "Yes" or "No",
    "Serious AE": "Yes" or "No"
  }}
]

Use these keys EXACTLY as written. 
Do NOT add question marks, extra spaces, or any additional keys.
Do NOT change capitalization.

Patient MRN: {{mrn}}
"""

# ✅ 保存结果
structured_results = []
results = []

# 统一清理一下列名（去掉空格）
df.columns = df.columns.str.strip()

mrn_col = "hash"
time_col = "Collected Date Time"
order_col = "Order Name"

# ✅ 按病人 + 采集时间 + Order Name 分组
group_cols = [mrn_col, time_col, order_col]

structured_results = []
results = []

for (mrn_value, collected_dt, order_name), chunk in df.groupby(group_cols, dropna=False):
    # 如果这组完全是空的也可以跳过（可选）
    if chunk.empty:
        continue

    # 拼接 lab 文本：逐行把所有列串起来
    lines = []
    for _, r in chunk.iterrows():
        line = "; ".join([f"{col}: {r[col]}" for col in df.columns])
        lines.append(line)
    lab_text = "\n".join(lines)

    # 填 prompt
    prompt = base_prompt.format(text=lab_text, mrn=mrn_value)

    print(f"\n=== Processing MRN: {mrn_value} | Collected: {collected_dt} | Order: {order_name} | n={len(chunk)} ===")

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=2048
        )
        reply = response.choices[0].message.content.strip()

        # 保存原始结果（方便以后 debug）
        results.append({
            "MRN": mrn_value,
            "Collected Date Time": collected_dt,
            "Order Name": order_name,
            "Lab Summary": lab_text,
            "AE Extracted": reply
        })

        # 解析 JSON
        jstart = reply.find("[")
        jend = reply.rfind("]") + 1
        jtext = reply[jstart:jend]

        if not (jtext.startswith("[") and jtext.endswith("]")):
            print("⚠️ Not valid JSON array.")
            continue

        ae_list = json.loads(jtext)
        if not isinstance(ae_list, list) or len(ae_list) == 0:
            print("ℹ️ No AE extracted.")
            continue

        for ae in ae_list:
            # 额外附上本组的 meta 信息（方便之后和 lab 表回对）
            ae["Group MRN"] = mrn_value
            ae["Collected Date Time"] = str(collected_dt)
            ae["Order Name"] = str(order_name)
            structured_results.append(ae)

    except Exception as e:
        print(f"❌ Error for MRN={mrn_value}, time={collected_dt}, order={order_name}: {e}")
        continue

    time.sleep(1)

# ✅ 保存输出
pd.DataFrame(results).to_csv("/data/gey2/Lab266632-1_04_1212_gpt4o_results.csv", index=False)
out_path = "/data/gey2/Lab266632-1_04_1212_gpt4o_structured_ae.csv"
pd.DataFrame(structured_results).to_csv(out_path, index=False)

print(f"\n✅ 完成，共写入 {len(structured_results)} 条结构化 AE -> {out_path}")

