# Operationalizing Externally Trained ML Models in Splunk MLTK via ONNX

**BGP Failure Prediction · Keras · tf2onnx · MLflow · Databricks · Splunk MLTK**

> Presented at the **Fort Worth Splunk User Group**, March 26, 2026  
> Speaker: **Abhiraj Malappa**, bitsIO  
> Event: [usergroups.splunk.com](https://usergroups.splunk.com/events/details/splunk-fort-worth-splunk-user-group-presents-operationalizing-externally-trained-ml-models-in-splunk-mltk/)

---

## What this repository is

Splunk MLTK is powerful — but it has three hard ceilings: a 90-day data retention limit, no support for neural networks or deep learning frameworks, and no MLOps governance. This repository is the complete working implementation of an architecture that gets past all three, without abandoning Splunk as the inference environment.

The use case is **BGP failure prediction**: predicting router BGP session failures 1–2 hours before they occur, using 14 features across three source systems (Splunk, SNMP, CMDB), a Keras neural network trained on Databricks, and real-time `| apply` inference in Splunk MLTK every 15 minutes.

The pattern generalises to any model, any framework, any Splunk environment.

---

## The architecture

```
Data Lake (6-month BGP history)
    │
    ▼
Databricks — train Keras neural network
    │  focal loss · class weights · 14 features
    ▼
MLflow — track experiment · register .onnx artifact
    │
    ▼  tf2onnx (opset 12)
    │
    ▼
bgp_failure_model.onnx ──► Splunk MLTK
                                │  | apply every 15 min
                                ▼
                         failure_probability per router
                                │
                                ▼
                    Drift alert fires ──► Webhook ──► Databricks retrain job
                                                           │
                                                           └──► loop closed
```

**Why ONNX?** The `.onnx` file is the only thing that crosses the boundary between the training world and the inference world. It does not care how the model was trained or where it runs. That is the entire point.

---

## Three reasons we train outside Splunk

| Reason | Detail |
|--------|--------|
| **Data** | 6-month training window required. Splunk retains 90 days. Historical data lives in the data lake. |
| **Model** | Keras neural network with focal loss and class weights. 14 features, non-linear interactions. MLTK cannot run this. |
| **Governance** | MLflow gives us experiment tracking, model versioning, drift baselines, and a full audit trail. MLTK has none of these. |

---

## Repository structure

```
├── bgp_prediction_notebook.ipynb   # Databricks training notebook — 33 cells
├── deploy_onnx_to_splunk.py        # REST API deployment script (L1/L2/L3)
├── bgp_scored_events.csv           # Sample data for Splunk lookup (240 rows)
├── bgp_inference_sample.csv        # Raw inference data with ground truth
├── bgp_mlops_dashboard.xml         # Splunk dashboard XML (17 panels)
└── README.md
```

---

## The training notebook

`bgp_prediction_notebook.ipynb` — 33 cells, runs on Databricks (Community Edition compatible for the training portion).

### What it does

1 - Setup and imports 
2 - **Feature documentation** — 14 features, 3 source categories, why each matters 
3 - Data assembly from three sources (Splunk export, SNMP, CMDB) 
4 - Rolling window feature engineering (4-hour windows, 15-min intervals) 
5 - Class imbalance handling — focal loss + class weights 
6 - Keras model — Dense 64→32→16, ReLU+Dropout, sigmoid output 
7 - Evaluation — confusion matrix, risk score distribution, 38-day router timeline 
8 - Threshold tuning via precision-recall curve (F2-optimised) 
9 - ONNX conversion via tf2onnx 
10 - **ONNX Runtime validation** — 5-check pre-import checklist 
11 - MLflow logging — params, metrics, .onnx artifact, feature schema. **REST API push to Splunk** — `PUSH_TO_SPLUNK` flag, L1/L2/L3 options 
12 -  Monitoring SPL (prediction drift, feature drift, ServiceNow ground truth, data quality, training-serving skew) 
16 - Retraining trigger logic — `should_retrain()` function 
17 - Standalone script generator 



## The 14 features

| Category | Source | Features |
|----------|--------|---------|
| **BGP Protocol** | Splunk (Cisco IOS syslog) | `bgp_state_changes_4h`, `neighbor_resets_4h`, `holdtime_remaining_pct`, `route_flaps_4h`, `bgp_prefixes_received`, `bgp_notification_sent_4h` |
| **Device Health** | SNMP / External | `cpu_utilization_avg`, `cpu_utilization_max`, `memory_utilization_avg`, `interface_errors_4h`, `packet_loss_pct` |
| **Operational Context** | CMDB / External | `uptime_days`, `log_count_error_4h`, `log_count_warning_4h` |

**8 features come from Splunk. 6 come from external systems.** This hybrid sourcing is the primary justification for training externally and using ONNX to bring the model back in.

> ⚠️ **Column order is the integration contract.** The ONNX model knows column positions, not field names. The order above is the training order. The Feature Variables field in Splunk MLTK must list them in exactly this order.

---

## ONNX conversion — the one detail that matters

```python
input_signature = [
    tf.TensorSpec([None, 14], tf.float32, name='float_input')
]

tf2onnx.convert.from_keras(
    model,
    input_signature=input_signature,
    output_path='bgp_failure_model.onnx',
    opset=12
)
```

`name='float_input'` is the field name Splunk MLTK will use. It must match exactly what you configure in the MLTK model registration dialog. A mismatch causes **silent wrong scores** — no error, just incorrect predictions.

### Validate before importing into Splunk — always

```python
sess = rt.InferenceSession('bgp_failure_model.onnx')

print(sess.get_inputs()[0].name)   # must be: float_input
print(sess.get_inputs()[0].shape)  # must be: [None, 14]
print(sess.get_inputs()[0].type)   # must be: tensor(float)
```

Step 3 (validation) is the most commonly skipped step and the #1 source of MLTK import failures.

---

## Splunk MLTK registration

**Settings → Machine Learning Toolkit → Import Model → ONNX**

| Field | Value |
|-------|-------|
| Model Name | `bgp_failure_model` |
| Feature Variables | `bgp_state_changes_4h neighbor_resets_4h holdtime_remaining_pct route_flaps_4h bgp_prefixes_received bgp_notification_sent_4h cpu_utilization_avg cpu_utilization_max memory_utilization_avg interface_errors_4h packet_loss_pct uptime_days log_count_error_4h log_count_warning_4h` |
| Target Variable | `failure_probability` |
| Model File | `bgp_failure_model.onnx` |

**Verify immediately after import:**

```spl
| makeresults
| eval bgp_state_changes_4h=1, neighbor_resets_4h=0,
       holdtime_remaining_pct=88, route_flaps_4h=0,
       bgp_prefixes_received=8800, bgp_notification_sent_4h=0,
       cpu_utilization_avg=32, cpu_utilization_max=48,
       memory_utilization_avg=54, interface_errors_4h=2,
       packet_loss_pct=0.1, uptime_days=90,
       log_count_error_4h=1, log_count_warning_4h=6
| apply bgp_failure_model
| table failure_probability
```

If it returns a number between 0 and 1 — the model is wired correctly.

---

## Automated deployment — REST API push

`deploy_onnx_to_splunk.py` implements three deployment levels:

| Level | Mechanism | When to use |
|-------|-----------|-------------|
| **L1 — Manual** | `mlflow artifacts download` + `cp` to MLTK lookups | First deploy / demo |
| **L2 — REST API push** | `requests.post()` to Splunk REST endpoint | Automated retraining loop |
| **L3 — CI/CD** | MLflow webhook → GitHub Actions → validate → push | Production gold standard |

```python
# The REST API call (L2)
requests.post(
    f"{SPLUNK_HOST}/servicesNS/nobody/Splunk_ML_Toolkit/data/lookup-table-files",
    headers={"Authorization": f"Bearer {TOKEN}"},
    files={"eai:data": ("bgp_failure_model.onnx", file_bytes, "application/octet-stream")},
)
```

`eai:data` is the Splunk REST API field name for file uploads. Non-negotiable.

**Run the deployment pipeline:**

```bash
SPLUNK_HOST=https://your-splunk:8089 \
SPLUNK_TOKEN=your-token \
MLFLOW_RUN_ID=your-run-id \
python deploy_onnx_to_splunk.py
```

---

## Splunk → Databricks retraining trigger (closing the loop)

Splunk fires a drift alert → alert action POSTs to Databricks Jobs API → retraining runs automatically.

```
POST https://<workspace>.azuredatabricks.net/api/2.0/jobs/run-now
Authorization: Bearer <databricks-token>
Content-Type: application/json

{"job_id": 12345, "notebook_params": {"trigger_reason": "prediction_drift"}}
```

**Splunk's built-in webhook action does not support custom headers.** Use either:
- **Splunkbase Webhook Alert Action** app (5-minute install)
- **Custom Python alert action** — `$SPLUNK_HOME/etc/apps/search/bin/trigger_databricks_retrain.py` (included in notebook Cell 16)

---

## Five monitoring types — all running in Splunk SPL

| # | Type | What it's asking |
|---|------|-----------------|
| 1 | **Prediction drift** | Is the model's output shifting without a real-world cause? |
| 2 | **Feature drift** | Has the world changed since we trained? |
| 3 | **Performance degradation** | Is the model actually right when it matters? |
| 4 | **Data quality** | Did the model receive valid inputs before scoring? |
| 5 | **Training-serving skew** | Are training and inference computing features the same way? |

> **Feature drift vs training-serving skew:**  
> Feature drift = the world changed (environment evolved — retrain on recent data).  
> Training-serving skew = your code disagrees with itself (pipeline divergence — fix the mismatch, then retrain).  
> One is about the data. The other is about the code.

---

## The Splunk dashboard

`bgp_mlops_dashboard.xml` — import directly into Splunk Classic Dashboards.

**17 panels across 7 rows:**
- KPI cards (Precision, Recall, F2 — colour-coded)
- Confusion matrix with business consequences per cell
- Per-router outcome summary (Caught / False alarm / Missed / No failure)
- Fleet risk timeline — `failure_probability` by router over time
- Risk score distribution
- Prediction drift with ±20% bounds
- Feature drift — live vs training baseline
- BGP signal trends
- Training-serving skew table
- Data quality gate
- Full scored event log

**To use:** Upload `bgp_scored_events.csv` as a Splunk lookup, then import the dashboard XML.

```
Settings → Lookups → Lookup table files → Add new
  Filename: bgp_scored_events.csv

Dashboards → Create New Dashboard → Classic → Source → paste XML
```

---

## Class imbalance — how we handle it

**0.5% positive rate (1 failure per 199 healthy windows)**

Two complementary techniques:

```python
# Focal loss — down-weights easy negatives during backpropagation
# gamma=2.0 focuses gradient on ambiguous pre-failure patterns
focal_loss(gamma=2.0, alpha=0.25)

# Class weights — encodes business cost ratio
# Missing a failure >> false alarm
class_weight = {0: 1.0, 1: 200.0}
```

**Accuracy is banned.** A model predicting all-healthy gets 99.5% accuracy. We use: **F2 score** (primary), precision, recall, average precision. F2 weights recall twice as heavily as precision — correct for a use case where missing a failure is catastrophically more expensive than a false alarm.

---

## Retraining triggers

The `should_retrain()` function in Cell 16 evaluates five conditions:

| Trigger | Threshold | Meaning |
|---------|-----------|---------|
| Prediction drift | >20% shift from baseline | Model output changing without cause |
| Feature drift | >25% shift from training mean | Environment has changed |
| Live F2 score | <0.65 | Performance has degraded below acceptable |
| Infrastructure event | Firmware upgrade / topology change | Known distribution shift |
| Time floor | 6 weeks since last retrain | Scheduled refresh regardless |

---

## Generalising the pattern

This architecture is not BGP-specific. The same pipeline works for:

| Use case | Model | ONNX export |
|----------|-------|-------------|
| Security threat classification | XGBoost | `onnxmltools` |
| IT ops resource forecasting | LSTM | `tf2onnx` |
| User behaviour anomaly | Autoencoder | `tf2onnx` |
| Log anomaly detection | Isolation Forest | `onnxmltools` |

The recipe is always the same: **train where your data is → govern with MLflow → export to ONNX → infer in Splunk**.

---

## Requirements

```
tensorflow>=2.10
tf2onnx>=1.14
onnxruntime>=1.16
onnx>=1.14
mlflow>=2.8
scikit-learn>=1.3
numpy>=1.24
pandas>=2.0
requests>=2.31
matplotlib>=3.7
```

---

## Citation / attribution

If you use this work, please cite the presentation:

```
Abhiraj Malappa (2026). Operationalizing Externally Trained ML Models in Splunk MLTK.
Fort Worth Splunk User Group, March 26, 2026.
https://usergroups.splunk.com/events/details/splunk-fort-worth-splunk-user-group-presents-operationalizing-externally-trained-ml-models-in-splunk-mltk/
```

---

## Author

**Abhiraj Malappa**  
bitsIO  
Fort Worth Splunk User Group presenter  

---

*"Train where your data is. Govern with MLflow. Deploy with ONNX. Infer in Splunk."*
