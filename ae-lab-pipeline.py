import time
import json
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from openai import AzureOpenAI

# ================= AzureGPT =================
client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint="https://bionlp-ge.openai.azure.com/",
    api_key=""
)

# ================= LAB  PROMPT =================
lab_prompt = """
You are a clinical research assistant helping to extract adverse events (AEs) from lab abnormalities.

Lab summary:
<text>
{text}
</text>

You MUST return ONLY a JSON array (no explanation). Format:

[
  {
    "MRN": "...",
    "Onset Date": "...",
    "Date Resolved": "...",
    "AE Term": "...",
    "Grade": "...",
    "Attribution to Disease": "...",
    "Immune-related AE": "Yes" or "No",
    "Serious AE": "Yes" or "No"
  }
]

Patient MRN: {mrn}
"""


# ================================================================
# Step 0 — GPT Extract AE from LAB
# ================================================================
def gpt_extract_lab(lab_csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(lab_csv_path).head(2000)
    df.columns = df.columns.str.strip()

    mrn_col = "hash"
    time_col = "Collected Date Time"
    order_col = "Order Name"

    group_cols = [mrn_col, time_col, order_col]

    structured = []

    for (mrn_value, collected_dt, order_name), chunk in df.groupby(group_cols):

        # 拼接文本
        lines = []
        for _, r in chunk.iterrows():
            line = "; ".join([f"{col}: {r[col]}" for col in df.columns])
            lines.append(line)
        lab_text = "\n".join(lines)

        prompt = lab_prompt.format(text=lab_text, mrn=mrn_value)

        print(f"\n=== LAB | MRN={mrn_value} | Time={collected_dt} | Order={order_name} ===")

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=2048
            )
            reply = response.choices[0].message.content.strip()

            js_start = reply.find("[")
            js_end = reply.rfind("]") + 1
            js = reply[js_start:js_end]

            if not js.startswith("["):
                print("⚠ LAB AE invalid JSON")
                continue

            ae_list = json.loads(js)
            if not isinstance(ae_list, list):
                continue

            for ae in ae_list:
                ae["MRN"] = mrn_value
                ae["Collected Date Time"] = collected_dt
                ae["Order Name"] = order_name
                ae["CTCAE"] = ae["AE Term"].lower().strip()
                structured.append(ae)

        except Exception as e:
            print("❌ LAB error:", e)

        time.sleep(1)

    df_lab = pd.DataFrame(structured)

    if not df_lab.empty:
        df_lab["Grade"] = pd.to_numeric(df_lab["Grade"], errors="coerce")

    print(f"✅ LAB AE extracted: {len(df_lab)} rows")
    return df_lab


# ================================================================
# Step 1 — Baseline filtering（可选）
# ================================================================
def filter_with_baseline(ae_df, baseline_file):
    if baseline_file in [None, ""] or ae_df.empty:
        print("ℹ️ No baseline filter applied")
        return ae_df

    baseline_df = pd.read_excel(baseline_file)
    baseline_df.columns = baseline_df.columns.str.strip()

    baseline_df["Patient"] = baseline_df["Patient"].astype(str).str.strip()
    baseline_df["Adverse Event Term (v5.0)"] = baseline_df["Adverse Event Term (v5.0)"].astype(str).str.lower()
    baseline_df["Grade"] = baseline_df["Grade"].astype(str).str.extract(r"(\d+)").astype(float)

    ae_df["MRN"] = ae_df["MRN"].astype(str)

    merged = ae_df.merge(
        baseline_df[["Patient", "Adverse Event Term (v5.0)", "Grade"]],
        left_on=["MRN", "CTCAE"],
        right_on=["Patient", "Adverse Event Term (v5.0)"],
        how="left"
    )

    baseline_grade = merged["Grade_y"].fillna(-1)
    keep = merged["Grade_x"] > baseline_grade

    filtered = merged[keep]
    print(f"✅ baseline filter: kept {len(filtered)} / {len(ae_df)}")

    return filtered[ae_df.columns]


# ================================================================
# Step 2 — MedCPT CTCAE mapping
# ================================================================
def map_to_ctcae_medcpt(ae_df, ctcae_dict_csv, medcpt_model_dir):
    if ae_df.empty:
        print("⚠ No AE to map")
        return ae_df

    ct = pd.read_csv(ctcae_dict_csv)
    ct["CTCAE"] = ct["CTCAE Term"].str.lower()

    terms = ct["CTCAE"].tolist()

    tokenizer = AutoTokenizer.from_pretrained(medcpt_model_dir)
    model = AutoModel.from_pretrained(medcpt_model_dir).eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    def encode_list(texts):
        out = []
        for t in texts:
            inputs = tokenizer(t, return_tensors="pt", truncation=True).to(device)
            with torch.no_grad():
                emb = model(**inputs).last_hidden_state[:, 0, :]
                emb = F.normalize(emb, dim=1)[0].cpu()
            out.append(emb)
        return torch.stack(out)

    ct_emb_cpu = encode_list(terms)

    mapped_terms = []
    for term in ae_df["CTCAE"]:
        inputs = tokenizer(term, return_tensors="pt", truncation=True).to(device)
        with torch.no_grad():
            emb = model(**inputs).last_hidden_state[:, 0, :]
            emb = F.normalize(emb, dim=1)

        sim = torch.mm(emb, ct_emb_cpu.to(device).T).squeeze()
        score, idx = torch.topk(sim, 1)

        mapped_terms.append(terms[idx])

    ae_df["CTCAE_Mapped_Top1"] = mapped_terms
    ae_df["Final_CTCAE_Term"] = mapped_terms

    return ae_df


# ================================================================
# Step 3 — Pipeline（LAB ONLY）
# ================================================================
def run_pipeline(
    lab_csv,
    baseline_xlsx,
    ctcae_dict_csv,
    medcpt_model_dir,
    final_output_csv,
):
    # Step 0: GPT Lab AE
    df_lab = gpt_extract_lab(lab_csv)

    # Step 1: Baseline filtering
    df_filter = filter_with_baseline(df_lab, baseline_xlsx)

    # Step 2: CTCAE mapping
    df_map = map_to_ctcae_medcpt(df_filter, ctcae_dict_csv, medcpt_model_dir)

    # Save
    df_map.to_csv(final_output_csv, index=False)
    print(f"\n🎉 Finished! Output saved to: {final_output_csv}")


# ================================================================
# Run example
# ================================================================
if __name__ == "__main__":

    LAB_CSV = "/netmnt/vast01/cbb01/lulab/gey2/AE_extraction/AE_Extraction_CTCAE_Map/Lab266632-1_04.csv"
    BASELINE = "/netmnt/vast01/cbb01/lulab/gey2/AE_extraction/AE_Extraction_CTCAE_Map/18C0056_BL_Subgroup_02.xlsx"
    CTCAE_DICT = "/netmnt/vast01/cbb01/lulab/gey2/AE_extraction/AE_Extraction_CTCAE_Map/CTCAE_v5.0.csv"
    MEDCPT_MODEL = "/netmnt/vast01/cbb01/lulab/gey2/AE_extraction/AE_Extraction_CTCAE_Map/medcpt_ctcae_triplet_epoch10"
    OUTPUT = "/netmnt/vast01/cbb01/lulab/gey2/AE_extraction/AE_Extraction_CTCAE_Map/lab_pipeline_output_test.csv"

    run_pipeline(
        lab_csv=LAB_CSV,
        baseline_xlsx=BASELINE,
        ctcae_dict_csv=CTCAE_DICT,
        medcpt_model_dir=MEDCPT_MODEL,
        final_output_csv=OUTPUT,
    )
