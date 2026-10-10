"""
FloodReady - Community Flood Early Warning & Relief Prioritisation
Mini Challenge Week 6 - Complete Notebook (run as a script or in Jupyter)
"""
import sys, io
# Force UTF-8 output on Windows so box-drawing characters display correctly
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────────────────────
# 0. IMPORTS
# ─────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

print("=" * 65)
print("  FloodReady – Flood Early Warning & Village Prioritisation")
print("=" * 65)

# ─────────────────────────────────────────────────────────────
# 1. PROBLEM STATEMENT
# ─────────────────────────────────────────────────────────────
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 1 – PROBLEM STATEMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Main User   : District Disaster Management Officer (DDMO)

Decisions the DDMO must make:
  1. Which river gauges will cross danger level within 48 hours?
  2. Which villages must be evacuated first?
  3. What alert channel (SMS / Loudspeaker / WhatsApp) for each village?
  4. When to mobilise relief teams and pre-position shelters?

Measurable Success Criteria:
  * Recall (flood detection) >= 0.90  -> miss as few real floods as possible
  * Precision >= 0.75                  -> limit unnecessary panic evacuations
  * Alert delivery < 30 minutes after danger threshold crossed
  * >= 95% of high-vulnerability villages receive alerts even with no phone coverage
  * Zero missed alerts for villages with shelter capacity < 50% of population
""")

# ─────────────────────────────────────────────────────────────
# 2. DATA INGESTION & QUALITY
# ─────────────────────────────────────────────────────────────
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("SECTION 2 – DATA INGESTION & QUALITY")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

# --- Load ---
readings = pd.read_csv("floodready_river_readings.csv", parse_dates=["Timestamp"])
stations = pd.read_csv("floodready_gauge_stations.csv")
villages = pd.read_csv("floodready_village_profiles.csv")

print(f"Readings shape : {readings.shape}")
print(f"Stations shape : {stations.shape}")
print(f"Villages shape : {villages.shape}\n")

# --- Missing Values ---
print("-- Missing Values --")
mv = readings.isnull().sum()
print(mv[mv > 0])
print()

# Fix: impute missing River_Level_m with median of same Gauge_ID
readings["River_Level_m"] = readings.groupby("Gauge_ID")["River_Level_m"] \
    .transform(lambda x: x.fillna(x.median()))
print("FIX: Missing River_Level_m imputed with per-gauge median.\n")

# --- Impossible Values (sentinel -99) ---
print("-- Impossible / Sentinel Values --")
impossible = readings[readings["River_Level_m"] < 0]
print(f"Rows with River_Level_m < 0 : {len(impossible)}")
print(impossible[["Reading_ID", "Gauge_ID", "River_Level_m"]])
# Action: flag and remove from training
readings["Sensor_Error"] = readings["River_Level_m"] < 0
readings_clean = readings[~readings["Sensor_Error"]].copy()
print(f"\nFIX: {len(impossible)} sensor-error row(s) flagged and excluded from modelling.\n")

# --- Duplicate Rows ---
print("-- Duplicate Rows --")
dups = readings_clean.duplicated(subset=["Gauge_ID", "Timestamp"])
print(f"Duplicate (Gauge_ID, Timestamp) pairs : {dups.sum()}")
readings_clean = readings_clean.drop_duplicates(subset=["Gauge_ID", "Timestamp"])
print("FIX: Duplicates removed.\n")

# --- Stuck Sensors (3+ identical River_Level_m in a row, same gauge) ---
print("-- Stuck Sensors --")
readings_clean = readings_clean.sort_values(["Gauge_ID", "Timestamp"])
readings_clean["Level_Same"] = readings_clean.groupby("Gauge_ID")["River_Level_m"] \
    .transform(lambda x: (x == x.shift(1)) & (x == x.shift(2)))
stuck = readings_clean[readings_clean["Level_Same"]]
print(f"Stuck-sensor rows detected : {len(stuck)}")
if len(stuck):
    print(stuck[["Reading_ID", "Gauge_ID", "Timestamp", "River_Level_m"]])
print("FLAG: Stuck-sensor rows flagged (kept but noted for officer review).\n")

# --- Join Datasets ---
print("-- Joining Datasets on Gauge_ID --")
merged = readings_clean.merge(stations, on="Gauge_ID", how="left")
full   = merged.merge(villages, on="Gauge_ID", how="left")
print(f"Merged dataset shape : {full.shape}")
print()

# ─────────────────────────────────────────────────────────────
# 3. FLOOD RISK PREDICTION
# ─────────────────────────────────────────────────────────────
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("SECTION 3 – FLOOD RISK PREDICTION")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

FEATURES = [
    "River_Level_m",
    "Upstream_Level_m",
    "Rainfall_mm",
    "Soil_Moisture_pct",
    "Warning_Level_m",
    "Danger_Level_m",
]
TARGET = "Flood_Within_48h"

# Deduplicate at gauge-timestamp level before modelling
model_df = merged.drop_duplicates(subset=["Gauge_ID", "Timestamp"]).copy()
model_df = model_df.dropna(subset=FEATURES + [TARGET])

# Time-based split (not random)
split_date = model_df["Timestamp"].quantile(0.75)
train_df = model_df[model_df["Timestamp"] <= split_date]
test_df  = model_df[model_df["Timestamp"] >  split_date]

X_train, y_train = train_df[FEATURES], train_df[TARGET]
X_test,  y_test  = test_df[FEATURES],  test_df[TARGET]

print(f"Train size : {len(X_train)} rows  (up to {split_date})")
print(f"Test  size : {len(X_test)}  rows  (after {split_date})\n")

# Train Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    class_weight={0: 1, 1: 3},   # penalise missed floods more
    random_state=42,
)
rf.fit(X_train, y_train)
y_pred  = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

print("-- Model Evaluation --")
print(classification_report(y_test, y_pred, target_names=["No Flood", "Flood"]))

prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test, y_pred, zero_division=0)
print(f"Precision : {prec:.2f}")
print(f"Recall    : {rec:.2f}")
print()
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print()

print("-- Feature Importance --")
fi = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
print(fi.to_string())
print()

print("-- Threshold Sensitivity --")
print(f"{'Threshold':>10}  {'Precision':>10}  {'Recall':>8}")
for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
    yp = (y_proba >= thresh).astype(int)
    p  = precision_score(y_test, yp, zero_division=0)
    r  = recall_score(y_test, yp, zero_division=0)
    print(f"{thresh:>10.1f}  {p:>10.2f}  {r:>8.2f}")
print("\nRecommended threshold = 0.40 (maximises recall, keeps precision acceptable)\n")

THRESHOLD = 0.40

# ─────────────────────────────────────────────────────────────
# 4. VILLAGE PRIORITISATION
# ─────────────────────────────────────────────────────────────
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("SECTION 4 – VILLAGE PRIORITISATION")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

def prioritise_villages(villages_df):
    df = villages_df.copy()
    df["vuln_ratio"]  = df["Vulnerable_Population"] / df["Population"]
    df["shortfall"]   = ((df["Population"] - df["Shelter_Capacity"]).clip(lower=0)
                         / df["Population"])
    df["dist_norm"]   = df["Distance_to_Shelter_km"] / df["Distance_to_Shelter_km"].max()
    df["road_flag"]   = (df["Road_Accessible_Flood"] == "No").astype(float)
    df["Priority_Score"] = (
        0.35 * df["vuln_ratio"]  +
        0.25 * df["shortfall"]   +
        0.20 * df["dist_norm"]   +
        0.20 * df["road_flag"]
    ) * 100
    df["Priority_Rank"] = df["Priority_Score"].rank(ascending=False).astype(int)
    return df.sort_values("Priority_Rank")

villages_ranked = prioritise_villages(villages)
print(villages_ranked[[
    "Village_Name", "Population", "Vulnerable_Population",
    "Shelter_Capacity", "Distance_to_Shelter_km",
    "Road_Accessible_Flood", "Priority_Score", "Priority_Rank"
]].to_string(index=False))
print()

# ─────────────────────────────────────────────────────────────
# 5 & 6. DECISION-READY ALERTS + PROTOTYPE
# ─────────────────────────────────────────────────────────────
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("SECTION 5 & 6 – ALERTS + PROTOTYPE OUTPUT")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

# Get latest reading per gauge
latest_readings = (
    readings_clean
    .sort_values("Timestamp")
    .groupby("Gauge_ID")
    .last()
    .reset_index()
)
latest_merged = latest_readings.merge(stations, on="Gauge_ID", how="left")

# Predict flood risk
X_latest = latest_merged[FEATURES].fillna(0)
latest_merged["Flood_Probability"] = rf.predict_proba(X_latest)[:, 1]
latest_merged["Flood_Alert"]       = (latest_merged["Flood_Probability"] >= THRESHOLD).astype(int)

# Join with village priority
villages_alerts = villages_ranked.merge(
    latest_merged[["Gauge_ID", "River_Level_m", "Danger_Level_m",
                   "Flood_Probability", "Flood_Alert", "Timestamp"]],
    on="Gauge_ID", how="left"
)
alert_villages = villages_alerts[villages_alerts["Flood_Alert"] == 1].copy()
alert_villages = alert_villages.sort_values("Priority_Rank")

def delivery_channel(row):
    if row["Road_Accessible_Flood"] == "No" or row["Phone_Coverage_pct"] < 40:
        return "Loudspeaker"
    elif row["Alert_Channel"] == "WhatsApp":
        return "WhatsApp"
    else:
        return "SMS"

alert_villages["Delivery"] = alert_villages.apply(delivery_channel, axis=1)

print("=" * 65)
print("  FloodReady -- DECISION-READY ALERT REPORT")
print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 65)

for _, row in alert_villages.iterrows():
    shortfall = max(0, int(row["Population"] - row["Shelter_Capacity"]))
    print(f"\nALERT #{int(row['Priority_Rank'])}  |  {row['Village_Name']}  (Gauge: {row['Gauge_ID']})")
    print(f"  WHAT & WHEN  : River at {row['River_Level_m']:.1f} m "
          f"(danger {row['Danger_Level_m']:.1f} m). "
          f"Flood probability {row['Flood_Probability']:.0%}. "
          f"Expected within 48 hours.")
    print(f"  WHY PRIORITY : {int(row['Vulnerable_Population'])} vulnerable people; "
          f"shelter shortfall {shortfall} people; "
          f"road {'INACCESSIBLE' if row['Road_Accessible_Flood']=='No' else 'accessible'} during flood.")
    print(f"  ACTION       : Evacuate {int(row['Population'])} residents "
          f"to shelter ({row['Distance_to_Shelter_km']} km away). "
          f"Pre-position {shortfall} extra capacity.")
    print(f"  CHANNEL      : {row['Delivery']}  (coverage {row['Phone_Coverage_pct']}%)")

print(f"\nTotal villages on alert : {len(alert_villages)}")
print("Awaiting officer approval before dispatch.\n")

# ─────────────────────────────────────────────────────────────
# 7. SYSTEM ARCHITECTURE
# ─────────────────────────────────────────────────────────────
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("SECTION 7 – SYSTEM ARCHITECTURE SUMMARY")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
print("""
Layers:
  1. SENSORS       : River gauge sensors + rainfall API feeds
  2. INGESTION     : Validate, deduplicate, flag outliers -> Time-series DB
  3. PREDICTION    : Random Forest model (retrained daily)
  4. ALERT ENGINE  : Prioritise villages, compose messages
  5. OFFICER UI    : Approve / modify / reject each alert batch
  6. DELIVERY      : SMS gateway | WhatsApp API | Loudspeaker (IVRS/truck)

Security  : HTTPS, role-based access, vulnerability data encrypted at rest
Scalability: Docker containers, Kubernetes for horizontal scaling
Fallback  : Sensor offline >2h -> flag to officer + use upstream readings
             Model confidence <60% -> officer manual confirmation required
""")

# ─────────────────────────────────────────────────────────────
# 9. RESPONSIBLE AI
# ─────────────────────────────────────────────────────────────
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("SECTION 9 – RESPONSIBLE AI")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
print("""
1. Recall-first threshold (0.40): missing a real flood is far costlier
   than a false alarm. Officers can raise threshold if alert fatigue occurs.

2. Equity: villages with phone coverage <40% or inaccessible roads auto-trigger
   loudspeaker/truck dispatch regardless of any other setting.

3. Privacy: vulnerability data (disability, income) encrypted; only officers
   can view individual village breakdowns.

4. Human in the loop: model outputs are recommendations only.
   Officer must approve every alert batch before messages are sent.

5. Fallback: static SMS/loudspeaker protocols activate if the system is down.
""")

print("=" * 65)
print("  End of FloodReady Notebook -- All sections complete.")
print("=" * 65)
