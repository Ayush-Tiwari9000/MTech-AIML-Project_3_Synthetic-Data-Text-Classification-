# Impact of Generative AI Synthetic Data on Text Classification with Limited Training Data

## 1. Project Overview

This project investigates whether synthetic labelled text generated using
Generative AI can compensate for a limited amount of real training data in
a text-classification task.

The project is implemented as **Assignment 9** of the Modern AI (AI26501)
course.

The text-classification task used in this project is **SMS spam detection**,
where messages are classified into two classes:

- `ham` - legitimate/non-spam message
- `spam` - spam message

A limited amount of real training data is used initially. Additional
synthetic training examples are then introduced in increasing amounts to
study their effect on classification performance.

---

## 2. Research Question

**Can Generative AI-generated synthetic labelled text compensate for limited
real training data in an SMS spam classification task?**

---

## 3. Hypothesis

Adding synthetic training examples to a limited real training dataset will
improve text-classification performance. As more synthetic data is added,
the model is expected to achieve better predictive performance, although the
improvement may eventually become smaller.

---

## 4. Dataset

The project uses the **SMS Spam Collection** dataset.

### Dataset Characteristics

- Total messages: **5,572**
- HAM messages: **4,825**
- SPAM messages: **747**
- Classification type: **Binary text classification**

The dataset was divided into:

- **80% training data**
- **20% test data**

A fixed random seed and stratified splitting were used to maintain
consistent class proportions.

### Limited Real-Data Setting

Only **20% of the available training portion** was retained as real
training data.

This produced:

- Real HAM training samples: **771**
- Real SPAM training samples: **120**
- Total limited real training samples: **891**

The test set contains **1,115 real messages** and is kept unchanged across
all experiments.

---

## 5. Synthetic Data

Additional labelled SMS messages were generated using a Generative AI
system (ChatGPT).

The final synthetic dataset contains:

- Synthetic HAM messages: **771**
- Synthetic SPAM messages: **120**
- Total synthetic messages: **891**

The synthetic messages are used only for training and are not included in the
test set.

---

## 6. Methodology
```text
The same machine-learning pipeline is used for every experiment:

SMS Text
   |
   v
TF-IDF Vectorization
   |
   v
Logistic Regression
   |
   v
HAM / SPAM Prediction
```
---

## 7. Experimental Conditions

Four training conditions were evaluated:

| Experiment | Real Training Data | Synthetic Data | Total Training Data |
|---|---:|---:|---:|
| Real Only | 891 | 0 | 891 |
| Real + 25% Synthetic | 891 | 223 | 1,114 |
| Real + 50% Synthetic | 891 | 446 | 1,337 |
| Real + 100% Synthetic | 891 | 891 | 1,782 |

The same real test set containing 1,115 messages was used for all four
experiments.

---

## 8. Evaluation Metrics

The experiments were evaluated using:

- **Accuracy**
- **Macro-F1**
- **Matthews Correlation Coefficient (MCC)**
- **Training Time**

Macro-F1 and MCC are particularly useful for evaluating performance when the
classes are imbalanced.

---

## 9. Results

| Experiment | Accuracy | Macro-F1 | MCC | Training Time (sec) |
|---|---:|---:|---:|---:|
| Real Only | 0.8744 | 0.5232 | 0.2297 | 0.0407 |
| Real + 25% Synthetic | 0.8816 | 0.5704 | 0.3168 | 0.0333 |
| Real + 50% Synthetic | 0.8888 | 0.6135 | 0.3856 | 0.0518 |
| Real + 100% Synthetic | **0.8933** | **0.6386** | **0.4234** | 0.0622 |

### Main Findings

Adding synthetic training data improved all three predictive performance
metrics in this experiment.

Compared with the real-only condition:

- Accuracy increased from **0.8744 to 0.8933**.
- Macro-F1 increased from **0.5232 to 0.6386**.
- MCC increased from **0.2297 to 0.4234**.

The best observed predictive performance was obtained with **Real + 100%
Synthetic Data**.

Training time remained small overall, although it increased for the largest
training condition.

---

## 10. Interesting / Failure Cases

The real-only baseline showed a strong difference between the two classes.

The baseline classification report showed:

- HAM recall: **1.00**
- SPAM recall: **0.06**
- HAM F1-score: **0.93**
- SPAM F1-score: **0.11**

This indicates that the baseline model strongly favoured the majority HAM
class and missed many SPAM messages.

After synthetic data was added, the overall Macro-F1 and MCC improved,
indicating better balanced classification performance.

---

## 11. Conclusion

The results support the hypothesis that synthetic labelled data can help
compensate for limited real training data in this SMS spam classification
experiment.

As synthetic data was added, Accuracy, Macro-F1 and MCC increased, with the
Real + 100% Synthetic condition producing the best observed performance.

Therefore, within this experiment, Generative AI-generated synthetic examples
provided useful additional training data when the amount of real training
data was restricted.

---

## 12. Limitations

- Only one text-classification dataset was used.
- Only one classifier was evaluated.
- Synthetic data quality depends on the generative process.
- The experiment used a relatively small restricted real training set.
- Results from one dataset should not automatically be generalized to all
  text-classification problems.

---

## 13. Future Improvements

Future work could investigate:

- Larger and more diverse datasets.
- Additional classifiers such as SVM or transformer-based models.
- Different synthetic-data generation strategies.
- Multiple Generative AI models.
- Different real/synthetic data ratios.
- Measures of synthetic-data quality and diversity.

---

## 14. Repository Structure
```text
MTech-AIML-Project_3_Synthetic-Data-Text-Classification/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── README.md
│   ├── sample_data.csv
│   ├── sms_spam.csv
│   └── synthetic_data.csv
│
├── notebooks/
│   └── experiment.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   └── evaluation.py
│
├── results/
│   ├── results.csv
│   ├── comparison_table.csv
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

Open a terminal in the project directory and run:

```bash
pip install -r requirements.txt
```

---
## 16. Reproducibility
A fixed random seed (random_state=42) is used for dataset splitting and synthetic-data subset selection.

The same real test set is used in all four experiments, while the amount of synthetic training data is changed between experimental conditions.

---

## 17. Project Summary
This project studies the practical use of Generative AI-generated synthetic data for improving text classification when labelled real data is limited.

The experimental results show that progressively adding synthetic training examples improved the observed Accuracy, Macro-F1, and MCC of the SMS spam classifier, with the highest values obtained using the Real + 100% Synthetic condition.