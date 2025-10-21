#Imports
from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Preprocessing
# fetch dataset 
default_of_credit_card_clients = fetch_ucirepo(id=350) 
  
# data (as pandas dataframes) 
X = default_of_credit_card_clients.data.features 
y = default_of_credit_card_clients.data.targets 

# Stores variable names
variables = default_of_credit_card_clients.variables

# Combines X and y for preprocessing
data = pd.concat([X, y], axis=1) 
# Labels the Variables with more useful names
variable_names = variables['description'].values
data.columns = variable_names[1:25]

# Filters out users who were billed nothing
data_new= data[data.iloc[:,11:17].any(axis=1)]

# Seperates the target variable
y_new = data_new['default payment next month']

# Seperates the X variables 
X_new = data_new.iloc[:, 0:23]

X_new['total_pay'] = X_new.iloc[:, 17:23].sum(axis=1) # The total amount payed across the six months
X_new['total_bill'] = X_new.iloc[:, 11:17].sum(axis=1) # The total amount billed across the six months
 # The total amount paid divided by the total amount billed across the six months
X_new['total_pay_to_total_bill'] = [p/b if b != 0 else 0 for p, b in zip(X_new['total_pay'], X_new['total_bill'])]

# Encodes Education with one hot encoding
education_encoded = pd.get_dummies(X_new['EDUCATION'], prefix='EDUCATION', dtype=float) 
X_new = pd.concat([X_new, education_encoded], axis=1) 
X_new = X_new.drop(['EDUCATION'], axis=1) # Drops original Education column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
X_new, y_new, test_size=0.2, random_state=42, stratify=y_new
)

# Initialize the scaler
scaler = StandardScaler() 

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test) 

# Initialize regression model
LRmodel = LogisticRegression(max_iter=15000)

#Fit the model
LRmodel.fit(X_train_scaled, np.ravel(y_train))

# Evaluates the accuracy
train_score = LRmodel.score(X_train_scaled, y_train)
test_score = LRmodel.score(X_test_scaled, y_test)

# Shows accuracy score
print("train: ")
print(train_score)
print("test: ")
print(test_score)

# Shows coefficients
print("Coefficents:")
print(LRmodel.coef_)

# Predict
y_pred = LRmodel.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

# Define class names
class_names = ['No Default', 'Default']

# Plot Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
