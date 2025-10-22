# CSC422_Midterm
# Predicting Credit Card Default Risk

## Project Description

This project aims to predict the likelihood of a credit card client defaulting on their payments using linear regression. Credit cards are widely used by many people worldwide, and understanding default risks helps companies make informed lending decisions.

I used linear regression due to its interpretability, which makes it easier to explain which features most impact the likelihood of default.

---

## Dataset Information

- **Dataset**: Default of Credit Card Clients Dataset  
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)  
- **Records**: 30,000 clients  
- **Features**: 23 (demographics, bill amounts, payments, etc.)  
- **Target Variable**: `default payment next month` (1 = default, 0 = no default)

---

## Setup & Installation

```bash
# Clone the repository
git clone https://github.com/nrhekman/CSC422_Midterm

# Install dependencies
pip install -r requirements.txt

# Run the model
python Midterm_src.py
```

---

##  Reproducing Results

To reproduce the results of this project:

- Ensure all required dependencies are installed as specified in `requirements.txt`.
- Run the `Midterm_src.py` script to:
  - Load and preprocess the dataset
  - Train a linear regression model
  - Display model coefficients
  - Output the accuracy score and confusion matrix

- You can also explore the analysis and model development steps in the `notebook.ipynb` file, which includes:
  - Exploratory data analysis
  - Feature selection
  - Model evaluation and insights

---

##  Key Findings

###  Coefficients
- **Age** had the greatest positive coefficient (**0.64**), meaning it had the strongest positive association with the likelihood of default.
- **Limit Balance** had the second greatest impact with a negative coefficient (**-0.25**), indicating higher credit limits were associated with lower default risk.

###  Accuracy
- The model consistently achieved around 81% accuracy.
- This represented an improvement of about 3% over the baseline model.

###  Confusion Matrix
- The model was more likely to produce false negatives than false positives.
  - This means it was more likely to flag someone who defaulted as someone who was not going to default, rather than the other way around.

---
