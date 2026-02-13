import pandas as pd

INPUT = "SRBH-20.csv"
OUTPUT = "srbh_processed.csv"

print("Loading dataset...")
df = pd.read_csv(INPUT, low_memory=False)

# -----------------------------
# Step 1: Identify label column
# -----------------------------
label_cols = [c for c in df.columns if " - " in c]

if len(label_cols) == 0:
    raise ValueError("ERROR: No attack label columns found!")

print("Found label columns:")
for c in label_cols:
    print("  ", c)

# -----------------------------
# Step 2: Convert one-hot labels → single class
# -----------------------------
def get_label(row):
    for col in label_cols:
        if row[col] == 1:
            return col
    return "000 - Normal"

df["Label"] = df.apply(get_label, axis=1)

# -----------------------------
# Step 3: Create numeric features from HTTP fields
# -----------------------------
df["req_len"] = df["request_http_request"].astype(str).str.len()
df["body_len"] = df["request_body"].astype(str).str.len()
df["cookie_len"] = df["request_cookie"].astype(str).str.len()
df["ua_len"] = df["request_user_agent"].astype(str).str.len()

df["has_sql_kw"] = df["request_http_request"].astype(str).str.contains(
    "select|union|drop|insert|sleep|or 1=1", case=False, regex=True
).astype(int)

df["has_cmd_kw"] = df["request_http_request"].astype(str).str.contains(
    ";|&&|\\|\\||`|cat |wget |curl |bash", regex=True
).astype(int)

df["has_traversal"] = df["request_http_request"].astype(str).str.contains(
    "\\.\\./", regex=True
).astype(int)

# -----------------------------
# Step 4: Binary target (attack vs normal)
# -----------------------------
df["Target"] = (df["Label"] != "000 - Normal").astype(int)

# -----------------------------
# Step 5: Keep only useful columns
# -----------------------------
features = [
    "req_len",
    "body_len",
    "cookie_len",
    "ua_len",
    "has_sql_kw",
    "has_cmd_kw",
    "has_traversal",
    "Target"
]

df = df[features].dropna()

df.to_csv(OUTPUT, index=False)

print("\nSaved:", OUTPUT)
print("Rows:", len(df))
print("Attacks:", df["Target"].sum())
print("Normal:", (df["Target"] == 0).sum())
