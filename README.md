# Deep Learning–Based Intrusion Detection System (DL-IDS) — Report



## 1. Executive summary



This report summarizes our work on designing, training and evaluating deep-learning models for intrusion detection on IoT/network traffic. Using the CIC-IoT2023 dataset and a range of deep architectures (MLP, LSTM, CNN, Autoencoder and hybrid AE-xLSTM-CNN), we compare classical baselines and deep models, address severe class imbalance, explore model compression via knowledge distillation, and identify a deployable hybrid that attains state-of-the-art performance on our test splits. Key takeaway: the AE-XLSTM-CNN hybrid produced the best results (Test Loss 0.0532, **Accuracy 98.18%, **F1 = 0.9802) on the held-out test set of 234,741 samples. 



---



## 2. Problem statement & objectives



Modern networks (and IoT ecosystems) face increasing automated attacks and frequent novel exploits. The problem this project addresses:



* Build a real-time, **robust, and **explainable DL-IDS that:



  1. Detects a wide range of attack types (DoS/DDoS, Mirai, Recon, MITM, Port scan, etc.).

  2. Achieves high detection recall (>95%) while keeping precision high (>90%).

  3. Operates with acceptable inference latency for online deployment (<100 ms target).

  4. Adapts to evolving/zero-day attacks and resists adversarial evasion.

  5. Enables model compression (knowledge distillation) for deployment on resource constrained devices.

     All objectives and constraints are derived from the project brief and dataset characteristics. 



---



## 3. Dataset: CIC-IoT2023 (summary & key statistics)



We used the CIC-IoT2023 dataset as the primary data source because it is IoT-focused, recent and diverse.



Key statistics (aggregated across dataset files):



* Total files: 169.

* Unique features (columns): 47.

* Total samples (combined): 46,686,579.

* Unique attack types: 34 (many DDoS variants, Mirai, Recon, etc.).

* Approx. class balance: ~97.6% malicious / 2.4% benign (severe imbalance). 



Why CIC-IoT2023?



* Realistic smart-home / IoT traffic, supports both binary & multiclass tasks, 47 behavioral features suitable for temporal and spatial learning (LSTM, CNN), and wide attack coverage for robust multiclass training. 



---



## 4. Preprocessing & experimental setup



Preprocessing pipeline



* Combine multiple files into unified table(s).

* Handle missing values and outliers (dataset-dependent rules).

* Categorical features → one-hot encoding.

* Numeric features → StandardScaler normalization.

* Windowing / sequencing for temporal models (sequence length used: 10 in hybrid experiments).

* Train / validation / test splits ensure attack classes for novelty experiments can be held out if required. 



Training environment & hyperparameters (representative)



* Optimizer: Adam.

* Loss: Categorical Cross-Entropy for classification; MSE or reconstruction loss for autoencoders.

* Batch sizes: tuned per model to balance GPU memory constraints (batch processing was used to mitigate RAM limits).

* Early stopping, dropout and regularization used where applicable. 



---



## 5. Models evaluated (overview)



We evaluated a broad set of models to identify trade-offs between complexity, accuracy and deployability:



1. Classical baselines



   * Random Forest (feature-based)

   * Mini-Batch KMeans (clustering baseline)

2. Deep / sequence / hybrid models



   * MLP / Dense (10+ layers — paper style MLP).

   * LSTM → Dense (stacked LSTM variants, including E-LSTM).

   * 1-D CNN and 2-D CNN variants (for sequence and transformed tabular layouts).

   * Autoencoder (stacked DAE) + dense classifier (combined reconstruction + classification loss).

   * AE-XLSTM-CNN (hybrid) — Autoencoder encoder for compression, extended LSTM for temporal modeling, CNN blocks for hierarchical feature extraction, and a classification head. 



Knowledge distillation experiments: teacher (larger LSTM stacks) → student (compact LSTM) with soft target KL loss plus hard cross-entropy; temperature T and α weighting were applied (representative T=4, α≈0.7). 



---



## 6. Representative architectures (short descriptions)



### 6.1 LSTM-Dense (compact & deep variants)



* Two stacked LSTM layers (e.g., 128 → 64 units) → dense SoftMax; dropout between layers.

* Good at sequential dependencies; larger variants reached higher accuracy at cost of params. 



### 6.2 Autoencoder-Dense



* Encoder: progressively smaller dense layers → bottleneck.

* Decoder: mirror of encoder.

* Combined loss: reconstruction (MSE) + classification (Cross-Entropy) with weighting factor α for trade-off. Used for anomaly-style detection and representation learning. 



### 6.3 1-D CNN



* Several Conv1D blocks (Conv → BN → ReLU → MaxPool → Dropout), flatten → dense classifier.

* Effective at local pattern extraction across sequences. 



### 6.4 AE-XLSTM-CNN (best performing hybrid)



1. Autoencoder encoder compresses input to latent (e.g., 64–128 dims).

2. Extended LSTM (stacked, with dropout) models long temporal context.

3. CNN residual blocks extract hierarchical local patterns.

4. Global pooling + classification head (linear → SoftMax for 34 classes).

Produced the best metrics across experiments. 



### 6.5. Knowledge distillation (teacher → student)



Motivation: Deployability requires small, fast models. We distilled a large LSTM teacher into a compact LSTM student.



Teacher (example): LSTM(128) → Dropout → LSTM(64) → Dense(64) → Dense(32) → logits.

Student (example): LSTM(32) → Dense(32) → logits.



Distillation objective:



* Loss_total = α * KL(soft_teacher_logits, soft_student_logits, T) + (1-α) * CE(student_logits, true_labels)

* Empirical settings: T = 4.0, α ≈ 0.7 (representative). 



Observations: Student models trade some accuracy for much lower inference cost; further multi-teacher distillation and architecture search are planned. 

---



## 7. Results — quantitative summary



> All reported results below are taken from our experimental runs (see IDS-DL.pdf for detailed plots and per-epoch logs). 



### 7.1 Baselines



* Random Forest: stable performance, useful precision/recall trend; limited by imbalance. (See per-batch metrics in slides.) 

* Mini-Batch KMeans (Clustering): Accuracy ~74.9%, Balanced Accuracy ~**35.9%, Weighted F1 ~**68.6% — struggled with rare classes. 



### 7.2 Deep models


| Model                   | Test Accuracy | Precision | Recall | F1                                      |
|-------------------------|--------------:|----------:|-------:|-----------------------------------------:|
| LSTM-Dense (1 LSTM layer)      |        0.8749 |    0.8847 | 0.8749 |                                   0.8708 |
| LSTM-Dense (2 LSTM layers)     |        0.9643 |    0.9634 | 0.9643 |                                   0.9627 |
| Autoencoder-Dense       |        0.5808 |    0.5354 | 0.5808 |                                   0.4905 |
| 1-D CNN                 |   ≈0.78 (see slides) |          |        |                                         |
| AE-XLSTM-CNN (hybrid)   |        0.9818 |          — |      — | 0.9802 (Loss 0.0532, Samples: 234,741)   |


---



## 8. Challenges, failure modes & lessons learned



1. Severe class imbalance: benign traffic proportion is very small; naive training leads to bias toward dominant attack classes. Techniques explored: class weighting, balanced sampling, anomaly detection approaches (AE). 

2. Memory constraints: initial experiments hit RAM/GPU limits; solution: micro-batching, careful data loaders and feature caching. 

3. Overfitting risk: highly complex architectures overfit tabular traffic; simpler architectures sometimes generalized better. Strong regularization and validation protocols are necessary. 



---



## 10. Future work & next steps



1. Multi-Teacher Knowledge Distillation: combine guidance from several high-performing teacher models to improve student generalization. 

2. Model compression & optimization: pruning, quantization (INT8), dynamic layer scaling and TFLite/ONNX export for edge deployment. 

3. Adversarial robustness: adversarial training and certified defenses to improve resistance to evasion. 

4. Explainability: integrate SHAP/LIME for per-alert interpretability to support security operators. 

5. Continual learning pipeline: automated triage for flagged novel events, human-in-loop validation, and periodic retraining to capture new attack classes. 



---



## 12. Conclusion



We conducted a comprehensive evaluation across classical and modern deep learning approaches for intrusion detection on a large IoT dataset. The hybrid AE-XLSTM-CNN architecture achieved the best trade-off between detection performance and robustness (98.18% accuracy; F1=0.9802). For real-world deployment, the next steps include knowledge distillation, model compression, adversarial hardening, and explainability integration. This line of work moves us toward production-ready, lightweight DL-IDS solutions suitable for high-volume IoT environments. 



---
