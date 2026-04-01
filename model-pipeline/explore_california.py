# %% [cell 1] — Imports & Load
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "historical_data/california_historical.csv"
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
df.head(3)

# %% [cell 2] — Basic Info
print(df.dtypes)
print("\nNull counts:")
print(df.isnull().sum().sort_values(ascending=False))

# %% [cell 3] — Target Variable: fire_detected_binary
print("Target distribution:")
print(df["fire_detected_binary"].value_counts())
print(f"\nFire rate: {df['fire_detected_binary'].mean()*100:.2f}%")
df["fire_detected_binary"].value_counts().plot(kind="bar", title="Fire Detected (0=No, 1=Yes)")
plt.show()

# %% [cell 4] — Drop/Fix Bad Columns
# -9999 sentinel values → NaN
df.replace(-9999, np.nan, inplace=True)

# Columns with too many nulls to be useful
null_pct = df.isnull().mean().sort_values(ascending=False)
print("Null % per column:")
print(null_pct[null_pct > 0])

# Drop columns that are IDs or leakage risks
drop_cols = ["grid_id", "region", "data_quality_flag", "date"]
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

# Parse timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.sort_values("timestamp").reset_index(drop=True)
print("\nDate range:", df["timestamp"].min(), "→", df["timestamp"].max())

# %% [cell 5] — Categorical Columns
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
print("Categorical columns:", cat_cols)

for col in cat_cols:
    print(f"\n{col} value counts:")
    print(df[col].value_counts().head(10))

# %% [cell 6] — Numeric Feature Distributions
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c != "fire_detected_binary"]

df[num_cols].hist(figsize=(20, 14), bins=40)
plt.suptitle("Numeric Feature Distributions", y=1.02)
plt.tight_layout()
plt.show()

# %% [cell 7] — Correlation with Target
corr = df[num_cols + ["fire_detected_binary"]].corr()["fire_detected_binary"].drop("fire_detected_binary")
corr_sorted = corr.abs().sort_values(ascending=False)
print("Feature correlation with fire_detected_binary:")
print(corr_sorted)

corr_sorted.plot(kind="barh", figsize=(10, 8), title="Abs Correlation with Target")
plt.tight_layout()
plt.show()

# %% [cell 8] — Fire vs No-Fire Feature Comparison
fire = df[df["fire_detected_binary"] == 1]
no_fire = df[df["fire_detected_binary"] == 0]

top_features = corr_sorted.head(8).index.tolist()
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for ax, col in zip(axes.flatten(), top_features):
    ax.hist(no_fire[col].dropna(), bins=40, alpha=0.5, label="No Fire", density=True)
    ax.hist(fire[col].dropna(), bins=40, alpha=0.5, label="Fire", density=True)
    ax.set_title(col)
    ax.legend(fontsize=7)
plt.suptitle("Top Feature Distributions: Fire vs No Fire")
plt.tight_layout()
plt.show()

# %% [cell 9] — Temporal Split Check
# Train = before 2025, Test = 2025 (LA fires)
df["year"] = df["timestamp"].dt.year
print("Rows per year:")
print(df.groupby("year")["fire_detected_binary"].agg(["count", "sum", "mean"]))

cutoff = pd.Timestamp("2025-01-01", tz="UTC")
train = df[df["timestamp"] < cutoff]
test = df[df["timestamp"] >= cutoff]
print(f"\nTrain: {len(train)} rows, fire rate: {train['fire_detected_binary'].mean()*100:.2f}%")
print(f"Test:  {len(test)} rows,  fire rate: {test['fire_detected_binary'].mean()*100:.2f}%")

# %% [cell 10] — Feature Engineering Checklist
print("=== PROCESSING SUMMARY ===")
print("\nCategorical → encode with LabelEncoder or OrdinalEncoder:")
print(cat_cols)

print("\nHigh-null columns → consider dropping or imputing:")
print(null_pct[null_pct > 0.1].index.tolist())

print("\nPotential leakage columns (fire info in features):")
leakage_candidates = [c for c in df.columns if any(x in c for x in ["fire", "frp", "confidence"])]
print([c for c in leakage_candidates if c != "fire_detected_binary"])

print("\nTarget: fire_detected_binary (binary classification)")
print("Recommended strategy: use scale_pos_weight in LightGBM due to class imbalance")

# %% [cell 11] — Quick Baseline Model
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

# Prep features
X_train = train.drop(columns=["fire_detected_binary", "timestamp", "year"])
X_test = test.drop(columns=["fire_detected_binary", "timestamp", "year"])
y_train = train["fire_detected_binary"]
y_test = test["fire_detected_binary"]

# Encode categoricals (LightGBM can handle them natively but needs int/category dtype)
for col in X_train.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))

# LightGBM handles NaN natively — no fillna needed
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight: {scale_pos_weight:.1f}  (handles class imbalance)")

clf = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=63,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
)
clf.fit(X_train, y_train, eval_set=[(X_test, y_test)])
y_prob = clf.predict_proba(X_test)[:, 1]

print("ROC-AUC:", roc_auc_score(y_test, y_prob).round(4))
print("PR-AUC (avg precision):", average_precision_score(y_test, y_prob).round(4))
print(classification_report(y_test, (y_prob > 0.5).astype(int)))

# %% [cell 12] — Feature Importance
feat_imp = pd.Series(clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)  # type: ignore
feat_imp.head(15).plot(kind="barh", figsize=(10, 6), title="Top 15 Feature Importances (RF Baseline)")
plt.tight_layout()
plt.show()
print(feat_imp.head(15))
