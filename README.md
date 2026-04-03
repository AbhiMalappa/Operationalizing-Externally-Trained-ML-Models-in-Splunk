# Operationalizing Externally Trained ML Models in Splunk MLTK via ONNX

**BGP Failure Prediction · Keras · tf2onnx · MLflow · Databricks · Splunk MLTK**

> Presented at the **Fort Worth Splunk User Group**, March 26, 2026  
> Speaker: **Abhiraj Malappa**, bitsIO  
> Event: [usergroups.splunk.com](https://usergroups.splunk.com/events/details/splunk-fort-worth-splunk-user-group-presents-operationalizing-externally-trained-ml-models-in-splunk-mltk/)

Article: [Medium](https://medium.com/@abhiraj7m/extending-splunk-ai-toolkit-deploying-an-externally-trained-deep-learning-model-in-splunk-using-0280c4d5664f)

## About

Splunk MLTK is powerful — but it has three hard ceilings: a 90-day data retention limit, no support for neural networks or deep learning frameworks, and no MLOps governance. This repository is the complete working implementation of an architecture that gets past all three, without abandoning Splunk as the inference environment.

The use case is **BGP failure prediction**: predicting router BGP session failures 1–2 hours before they occur, using 14 features across three source systems (Splunk, SNMP, CMDB), a Keras neural network trained on Databricks, and real-time `| apply` inference in Splunk MLTK every 15 minutes.

The pattern generalises to any model, any framework, any Splunk environment.


## The architecture

```
Data Lake (6-month BGP history)
      ↓
Train outside Splunk (Databricks + Keras)
      ↓
Track with MLflow (versioning, artifact registry)
      ↓
Export to ONNX (.onnx file)
      ↓
Import into Splunk AITK
      ↓
real-time inference ( | apply in SPL)
      ↓
Monitor drifts ──> Webhook ──> Databricks retrain job
      ↓
 loop closed
```

**Why ONNX?** The `.onnx` file is the only thing that crosses the boundary between the training world and the inference world. It does not care how the model was trained or where it runs. That is the entire point.

## Three reasons we train outside Splunk - 
* Data ceiling - AITK trains on data already in Splunk. Production ML models often need months of historical data to train on. Splunk's retention policies do not always support this. Your training data may live in a data lake outside Splunk's reach. 
* Algorithm ceiling - AITK supports classical ML. The moment you need fine-grained control - custom loss functions, or non-standard architectures, you are outside of what AITK can run natively. It cannot run Keras, PyTorch, or TensorFlow models.
* Governance ceiling - AITK has no experiment tracking, no model versioning, and no automated retraining triggers. A model deployed to AITK today cannot be reproducibly traced, compared, or retrained without significant manual effort.

## Repository structure

```
├── onnx_demo_notebook.ipynb        # Databricks training notebook 
├── deploy_onnx_to_splunk.py        # REST API deployment script (L1/L2/L3)
├── bgp_scored_events.csv           # Sample data for Splunk lookup (240 rows)
├── bgp_inference_sample.csv        # Raw inference data with ground truth
├── bgp_mlops_dashboard.xml         # Splunk dashboard XML (17 panels)
└── README.md
```

---

## The training notebook - What it does

1. Setup and imports
2. **Feature documentation** — 14 features, 3 source categories, why each matters
3. Data assembly from three sources (Splunk export, SNMP, CMDB)
4. Rolling window feature engineering (4-hour windows, 15-min intervals)
5. Class imbalance handling — focal loss + class weights
6. Keras model — Dense 64→32→16, ReLU+Dropout, sigmoid output
7. Evaluation — confusion matrix, risk score distribution, 38-day router timeline
8. Threshold tuning via precision-recall curve (F2-optimised)
9. ONNX conversion via tf2onnx
10. **ONNX Runtime validation** — 5-check pre-import checklist
11. MLflow logging — params, metrics, .onnx artifact, feature schema. **REST API push to Splunk** — `PUSH_TO_SPLUNK` flag, L1/L2/L3 options
12. Monitoring SPL (prediction drift, feature drift, ServiceNow ground truth, data quality, training-serving skew)
13. Retraining trigger logic — `should_retrain()` function 


## Splunk MLTK Automated deployment — REST API push

| Level | Mechanism | When to use |
|-------|-----------|-------------|
| L1 — Manual | `mlflow artifacts download` + `cp` to MLTK lookups | First deploy / demo |
| L2 — REST API push | `requests.post()` to Splunk REST endpoint | Automated retraining loop |
| L3 — CI/CD | MLflow webhook → GitHub Actions → validate → push | Production gold standard |


## Splunk → Databricks retraining trigger (closing the loop)

Splunk fires a drift alert -> alert action POSTs to Databricks Jobs API-> retraining runs automatically.


## Five monitoring types — all running in Splunk SPL


1. **Prediction drift** - Is the model's output shifting without a real-world cause? 
2. **Feature drift** - Has the world changed since we trained? 
3. **Performance degradation** - Is the model actually right when it matters? 
4. **Data quality** - Did the model receive valid inputs before scoring? 
5. **Training-serving skew** - Are training and inference computing features the same way? 


## Generalising the pattern

This architecture is not BGP-specific. The same pipeline works for:

| Use case                       | Model                  | Export tool   |
|--------------------------------|------------------------|---------------|
| Security threat classification | XGBoost                | `onnxmltools` |
| Capacity forecasting           | LSTM                   | `tf2onnx`     |
| Log anomaly detection          | Isolation Forest       | `onnxmltools` |
| Any sklearn model              | RandomForest, SVM, etc | `sklearn-onnx`|


The recipe is always the same: **train where your data is → govern with MLflow → export to ONNX → infer in Splunk**.

---

## Author

**Abhiraj Malappa**  
bitsIO  
Fort Worth Splunk User Group presenter  

---

*"Train where your data is. Govern with MLflow. Deploy with ONNX. Infer in Splunk."*
