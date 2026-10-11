# Data Directory

This directory contains the real SMS dataset, a small sample of the data, and the synthetic dataset generated for the project.

## Files

### 1. `sms_spam.csv`

The original SMS Spam Collection dataset, converted into CSV format.

- Total messages: 5,572
- HAM messages: 4,825
- SPAM messages: 747
- Columns: `text`, `label`

The `label` column contains two classes: `ham` and `spam`.

Source: [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)

### 2. `sample_data.csv`

A small sample of SMS messages provided to demonstrate the dataset format. It is not the full dataset used in the experiment.

### 3. `synthetic_data.csv`

Contains 891 labelled SMS messages generated using ChatGPT for the experiment.

- Synthetic HAM messages: 771
- Synthetic SPAM messages: 120
- Total synthetic messages: 891

These examples are used to augment the restricted real training data.

## Data Preparation

The real dataset is split into 80% training data and 20% testing data using stratified sampling and a fixed random seed.

Only 20% of the training portion is retained as real training data. Synthetic examples are then added at 25%, 50%, and 100% of the restricted real training-set size.

The same real test set is used for all experiments. Synthetic examples are used only for training and are not added to the test set.