# Data Directory

This directory contains the cleaned SMS Spam Collection dataset and the synthetic SMS dataset generated using Generative AI (ChatGPT) for the text-classification experiment.

---

## Dataset Files

### 1. `sms_spam.csv`
The SMS Spam Collection dataset, rigorously cleaned and deduplicated.

- **Source**: [UCI Machine Learning Repository - SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
- **Format**: CSV (`text`, `label`)
- **Classes**: `ham` (legitimate) and `spam`
- **Total Cleaned Messages**: 5,157
  - `ham`: 4,516 messages (87.57%)
  - `spam`: 641 messages (12.43%)

#### Cleaning & Preprocessing Applied
- **De-duplication**: 415 duplicate/redundant rows were identified and removed (reducing from the raw 5,572 down to 5,157 unique messages).
- **Unicode Normalization (NFKD)**: Smart punctuation (typographical single and double quotes, dashes, ellipses, non-breaking spaces) standardized to standard ASCII equivalents.
- **Whitespace Sanitization**: Stripped leading/trailing whitespace and collapsed redundant internal spaces.
- **Label Validation**: Enforced lowercase normalization with strict validation for `ham` and `spam`.

---

### 2. `synthetic_data.csv`
Contains labelled synthetic SMS messages generated using ChatGPT to augment the limited real training set.

- **Format**: CSV (`text`, `label`)
- **Total Cleaned Messages**: 888
  - Synthetic `ham`: 768 messages
  - Synthetic `spam`: 120 messages
- **Cleaning Applied**:
  - Unicode quotation normalization (curly quotes converted to ASCII apostrophes).
  - Indian Rupee (₹) symbol converted to `Rs `.
  - Whitespace trimming and deduplication.

---

## Experimental Data Partitioning

1. **Test Set Split**:
   - The cleaned dataset (5,157 samples) is split into **80% training** (4,125 samples) and **20% testing** (1,032 samples) using stratified sampling (`random_state=42`).
   - The test set contains **904 HAM** and **128 SPAM** messages and remains completely fixed across all experiments.

2. **Limited Real-Data Setting**:
   - To simulate a low-resource scenario, only **20% of the training portion** is retained as real training data.
   - This yields **825 limited real training messages**:
     - **722 HAM**
     - **103 SPAM**

3. **Synthetic Augmentation Conditions**:
   - Synthetic data subsets are nested and added at proportions of the limited real data size:
     - **25% Synthetic**: +206 samples (180 ham, 26 spam) &rarr; 1,031 total training
     - **50% Synthetic**: +413 samples (361 ham, 52 spam) &rarr; 1,238 total training
     - **100% Synthetic**: +825 samples (722 ham, 103 spam) &rarr; 1,650 total training
   - Synthetic data is used **strictly for training** and never appears in the evaluation test set.
