# Impact of Generative AI Synthetic Data on Text Classification with Limited Training Data

## 1. Project Overview

This project investigates whether synthetic labelled text generated using Generative AI can compensate for a limited amount of real training data in a text-classification task.

The project is implemented as **Assignment 9** of the Modern AI (AI26501) course.

The text-classification task used in this project is **SMS spam detection**, where messages are classified into two classes:

- `ham` - legitimate/non-spam message
- `spam` - spam message

A limited amount of real training data is used initially to simulate a low-resource scenario. Additional synthetic training examples are then introduced in increasing amounts (25%, 50%, and 100% of the restricted real training size) to study their effect on classification performance.

---

## 2. Research Question

**Can Generative AI-generated synthetic labelled text compensate for limited real training data in an SMS spam classification task?**

---

## 3. Hypothesis

Adding synthetic training examples to a limited real training dataset will improve text-classification performance. As more synthetic data is added, the model is expected to achieve better predictive performance (higher Accuracy, Macro-F1, and MCC).

---

## 4. Dataset & Data Cleaning

The project uses the **SMS Spam Collection** dataset from the UCI Machine Learning Repository.

### Data Cleaning & Preprocessing
Prior to experimentation, the raw data underwent thorough cleaning:
- **Deduplication**: 415 duplicate and redundant messages were removed, reducing data leakage and artificial frequency bias.
- **Unicode Normalization (NFKD)**: Typographical curly quotes, em/en dashes, ellipses, non-breaking spaces, and currency symbols were converted to standard ASCII equivalents.
- **Whitespace Sanitization**: Leading, trailing, and repeated whitespace characters were collapsed and trimmed.
- **Label Validation**: Enforced lowercase binary labels (`ham`, `spam`).

### Cleaned Dataset Characteristics

- Total messages: **5,157**
- HAM messages: **4,516** (87.57%)
- SPAM messages: **641** (12.43%)
- Classification type: **Binary text classification**

The cleaned dataset was partitioned into:
- **80% training data** (4,125 messages)
- **20% test data** (1,032 messages)

A fixed random seed (`random_state=42`) and stratified splitting were used to maintain consistent class proportions.

### Limited Real-Data Setting (Low-Resource Simulation)

Only **20% of the available training portion** was retained as real training data:

- Real HAM training samples: **722**
- Real SPAM training samples: **103**
- Total limited real training samples: **825**

The held-out test set contains **1,032 real messages** (904 HAM, 128 SPAM) and is kept strictly unchanged across all experiments.

---

## 5. Synthetic Data

Additional labelled SMS messages were generated using a Generative AI system (ChatGPT) and cleaned to match the normalization of the real dataset.

The cleaned synthetic dataset contains:

- Synthetic HAM messages: **768**
- Synthetic SPAM messages: **120**
- Total synthetic messages: **888**

The synthetic messages are used **only for training** and are never included in the test set.

---

## 6. Methodology

```text
SMS Text Message
       │
       ▼
Data Cleaning & Normalization (NFKD, whitespace, ASCII mapping)
       │
       ▼
TF-IDF Vectorization (Unigrams + Bigrams, English Stop-Words)
       │
       ▼
Logistic Regression Classifier (max_iter=1000, random_state=42)
       │
       ▼
Prediction (HAM / SPAM)
```

The identical machine-learning pipeline is evaluated across all four experimental conditions on the same fixed real test set.

---

## 7. Experimental Conditions

Four training conditions were evaluated:

| Experiment | Real Training Data | Synthetic Data | Total Training Data |
|---|---:|---:|---:|
| **Real Only** | 825 | 0 | 825 |
| **Real + 25% Synthetic** | 825 | 206 | 1,031 |
| **Real + 50% Synthetic** | 825 | 413 | 1,238 |
| **Real + 100% Synthetic** | 825 | 825 | 1,650 |

The same real test set containing 1,032 messages was used for all four experiments.

---

## 8. Evaluation Metrics

The experiments were evaluated using:

- **Accuracy**: Overall proportion of correctly classified messages.
- **Macro-F1**: Unweighted mean of F1-scores across both classes, critical for imbalanced datasets.
- **Matthews Correlation Coefficient (MCC)**: Balanced measure taking true and false positives and negatives into account.
- **Training Time**: Measured precisely via `time.perf_counter()`.

---

## 9. Results

| Experiment | Accuracy | Macro-F1 | MCC | Training Time (sec) |
|---|---:|---:|---:|---:|
| **Real Only** | 0.8798 | 0.4982 | 0.1658 | 0.0264 |
| **Real + 25% Synthetic** | 0.8828 | 0.5205 | 0.2196 | 0.0287 |
| **Real + 50% Synthetic** | 0.8847 | 0.5348 | 0.2493 | 0.0308 |
| **Real + 100% Synthetic** | **0.8905** | **0.5755** | **0.3227** | 0.0513 |

### Main Findings

Adding synthetic training data monotonically improved all predictive performance metrics:

- **Accuracy** increased from **0.8798 to 0.8905** (+1.07%).
- **Macro-F1** increased from **0.4982 to 0.5755** (+15.5% relative gain).
- **MCC** nearly doubled, rising from **0.1658 to 0.3227** (+94.6% relative gain).
- The best predictive performance was achieved by **Real + 100% Synthetic Data**.
- Training time remained negligible (<0.06 seconds for all conditions).

---

## 10. Interesting / Failure Cases

The baseline model trained solely on limited real data suffered heavily from class imbalance:

- **Baseline HAM Recall**: 1.00 (F1: 0.94)
- **Baseline SPAM Recall**: 0.03 (F1: 0.06)

Due to the heavy majority of HAM messages in the real data, the baseline classifier predicted the majority class almost exclusively, detecting only 4 out of 128 SPAM messages.

As synthetic data was added:
- SPAM recall rose from **0.0312 to 0.1172** (nearly 4x improvement).
- SPAM F1 rose from **0.0606 to 0.2098**.
- HAM classification remained virtually perfect (recall 1.00, precision 0.89).

---

## 11. Conclusion

The experimental results strongly support the hypothesis:
1. **Generative AI synthetic data effectively compensates for limited real data** when labelled training resources are constrained.
2. Progressive addition of synthetic examples (25% &rarr; 50% &rarr; 100%) produces consistent, monotonic gains in Accuracy, Macro-F1, and MCC.
3. Augmenting low-resource NLP pipelines with LLM-generated synthetic text is an efficient, cost-effective strategy for boosting classifier generalization.

---

## 12. Limitations

- A single text-classification task (SMS spam) was evaluated.
- Synthetic data was generated with ChatGPT prompts; diversity depends on prompt engineering.
- Standard TF-IDF + Logistic Regression baseline; deep learning / transformer models were not tested.

---

## 13. Future Improvements

- Evaluate transformer-based architectures (DistilBERT, RoBERTa).
- Explore class-weighted loss functions (`class_weight="balanced"`).
- Test diverse prompting strategies and temperature scaling during synthetic generation.
- Cross-domain generalization to other NLP datasets.

---

## 14. Repository Structure

```text
MTech-AIML-Project_3_Synthetic-Data-Text-Classification/
│
├── .gitignore                      # Git ignore rules
├── README.md                       # Comprehensive project documentation
├── requirements.txt                # Python package dependencies
│
├── data/
│   ├── README.md                   # Data documentation & cleaning methodology
│   ├── sms_spam.csv                # Cleaned real SMS Spam Collection dataset
│   └── synthetic_data.csv          # Cleaned ChatGPT-generated synthetic dataset
│
├── notebooks/
│   └── experiment.ipynb            # Interactive Jupyter notebook with complete pipeline
│
├── src/
│   ├── __init__.py                 # Python package initialization
│   ├── preprocessing.py            # Data loading, cleaning, and partitioning functions
│   ├── model.py                    # TF-IDF + Logistic Regression pipeline
│   └── evaluation.py               # Evaluation metrics and timing utilities
│
├── results/
│   ├── results.csv                 # Detailed metrics across all conditions
│   ├── comparison_table.csv        # Formatted comparison summary
│   └── figures/
│       ├── accuracy_comparison.png
│       ├── macro_f1_comparison.png
│       ├── mcc_comparison.png
│       └── training_time_comparison.png
│
└── presentation/
    └── Group3_Synthetic_Data_Text_Classification.pptx
```

---

## 15. How to Run

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Experiment via Script

```bash
python -c "import src; df=src.load_dataset(); print('Loaded', len(df), 'messages')"
```

Or open and run the notebook:

```bash
jupyter notebook notebooks/experiment.ipynb
```

---

## 16. Reproducibility

- Stratified data splitting with `random_state=42` ensures deterministic training and test splits.
- Synthetic subset selection is fixed via `random_state=42`.
- The same real test set (1,032 samples) is evaluated across all four experimental conditions.
