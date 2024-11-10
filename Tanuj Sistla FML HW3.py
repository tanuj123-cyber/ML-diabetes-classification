#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Numerical/scientific computing packages.
import numpy as np
import pandas as pd
import scipy

# Machine learning package.
import sklearn
from sklearn import metrics
from sklearn import model_selection 
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, classification_report, roc_auc_score, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_curve
from sklearn import datasets
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import make_pipeline

# Plotting packages.
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('diabetes.csv')
df = df.dropna()


# In[3]:


X = df.drop(['Diabetes'], axis = 1)
y = df['Diabetes']
X


# In[4]:


# Showing all data is linearly separable
for col in X:
    plt.scatter(df[col], y, c=y, cmap='bwr', alpha=0.5)
    plt.xlabel(col)
    plt.ylabel('diabetes')
    plt.title(f'Scatter Plot of {col} vs. Diabetes')
    plt.show()


# # Q1

# In[5]:


# Split the data into training and testing sets for each predictor
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
xTrainScaled = scaler.fit_transform(xTrain)
xTestScaled = scaler.transform(xTest)

logist = LogisticRegression(class_weight='balanced')

logist.fit(xTrainScaled, yTrain)

betas = logist.coef_[0]
print("Betas corresponding to each predictor:\n")
for pred, beta in zip(X.columns, betas):
    print(f"{pred}: {beta}")

yPred = logist.predict(xTestScaled)
yProb = logist.predict_proba(xTestScaled)[:, 1]

print(classification_report(yTest, yPred))
print("ROC AUC for full logistic regression model:", roc_auc_score(yTest, yProb))
matr = confusion_matrix(yTest, yPred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = matr, display_labels = [False, True])
cm_display.plot()
plt.show()


# In[6]:


false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(yTest, yProb)
plt.subplots(1, figsize=(10,10))
plt.title('ROC Curve for Logistic Regression (Full Model)')
plt.plot(false_positive_rate1, true_positive_rate1)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[7]:


minCol = ''
minroc = 100
for col in X:
    predictors = X.drop([col],axis=1)
    # Split the data into training and testing sets for each predictor
    xTrain, xTest, yTrain, yTest = train_test_split(predictors, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    xTrainScaled = scaler.fit_transform(xTrain)
    xTestScaled = scaler.transform(xTest)

    logist = LogisticRegression(class_weight='balanced')

    logist.fit(xTrainScaled, yTrain)
 
    yPred = logist.predict(xTestScaled)
    yProb = logist.predict_proba(xTestScaled)[:, 1]
    roc_score = roc_auc_score(yTest, yProb)
    if roc_score < minroc:
        minroc = roc_score
        minCol = col
        false_positive_rate, true_positive_rate, threshold = roc_curve(yTest, yProb)
    #Classifying and displaying target model prediction
    print(f"Classification Report for logistic regression model without {col}:\n", classification_report(yTest, yPred))
    print(f"ROC AUC for full logistic regression model without {col}:", roc_score)
    print("---------------------------------------------------------------------------") 


# In[8]:


print(f"The removal of {minCol} from the full model decreased the ROC AUC score to {minroc}, the most \ncompared to the removal of other predictors.")

plt.subplots(1, figsize=(10,10))
plt.title(f'ROC Curve for Logistic Regression (Full Model without ' + minCol + ')')
plt.plot(false_positive_rate, true_positive_rate)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# ## Q2

# In[9]:


# Split the data into training and testing sets for each predictor
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
xTrainScaled = scaler.fit_transform(xTrain)
xTestScaled = scaler.transform(xTest)

lsvc = LinearSVC(class_weight='balanced', max_iter=1000, dual=False, random_state=42)

lsvc.fit(xTrainScaled, yTrain)

betas = lsvc.coef_[0]
print("Betas corresponding to each predictor:\n")
for pred, beta in zip(X.columns, betas):
    print(f"{pred}: {beta}")

yPred = lsvc.predict(xTestScaled)
yProb = lsvc.decision_function(xTestScaled)
#Classifying and displaying target model prediction
print(classification_report(yTest, yPred))

print("ROC AUC for full SVM model:", roc_auc_score(yTest, yProb))
matr = confusion_matrix(yTest, yPred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = matr, display_labels = [False, True])
cm_display.plot()
plt.show() 


# In[10]:


false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(yTest, yProb)
plt.subplots(1, figsize=(10,10))
plt.title('ROC Curve for LinearSVC (Full Model)')
plt.plot(false_positive_rate1, true_positive_rate1)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[11]:


minCol = ''
minroc = 100
for col in X:
    predictors = X.drop([col],axis=1)
    
    # Split the data into training and testing sets for each predictor
    xTrain, xTest, yTrain, yTest = train_test_split(predictors, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    xTrainScaled = scaler.fit_transform(xTrain)
    xTestScaled = scaler.transform(xTest)

    lsvc = LinearSVC(class_weight='balanced', max_iter=1000, dual=False, random_state=42)

    lsvc.fit(xTrainScaled, yTrain)

    yPred = lsvc.predict(xTestScaled)
    yProb = lsvc.decision_function(xTestScaled)
    roc_score = roc_auc_score(yTest, yProb)
    if roc_score < minroc:
        minroc = roc_score
        minCol = col
        false_positive_rate, true_positive_rate, threshold = roc_curve(yTest, yProb)
        
    #Classifying and displaying target model prediction
    print(f"Classification Report for SVM model without {col}:\n", classification_report(yTest, yPred))
    print(f"ROC AUC for full SVM model without {col}:", roc_score)
    print("---------------------------------------------------------------------------") 


# In[12]:


print(f"The removal of {minCol} from the full model decreased the ROC AUC score to {minroc}, the most \ncompared to the removal of other predictors.")

plt.subplots(1, figsize=(10,10))
plt.title(f'ROC Curve for LinearSVC (Full Model without {minCol})')
plt.plot(false_positive_rate, true_positive_rate)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# # Q3

# In[15]:


# Split the data into training and testing sets for each predictor
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
xTrainScaled = scaler.fit_transform(xTrain)
xTestScaled = scaler.transform(xTest)

tree = DecisionTreeClassifier(random_state=42,class_weight='balanced')

tree.fit(xTrainScaled, yTrain)

yProb = tree.predict_proba(xTestScaled)[:, 1]
yPred = tree.predict(xTestScaled)

# Retrieve the feature importances
importances = tree.feature_importances_

# Print the feature importances
print("Feature Importances:")
for pred, importance in zip(X.columns, importances):
    print(f"{pred} (Importance): {importance:.4f}")

print(classification_report(yTest, yPred))

print("ROC AUC for full model:", roc_auc_score(yTest, yProb))
matr = confusion_matrix(yTest, yPred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = matr, display_labels = [False, True])
cm_display.plot()
plt.show()


# In[16]:


false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(yTest, yProb)
plt.subplots(1, figsize=(10,10))
plt.title('ROC Curve for Decision Tree (Full Model)')
plt.plot(false_positive_rate1, true_positive_rate1)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[17]:


minCol = ''
minroc = 100
for col in X:
    predictors = X.drop([col],axis=1)

    # Split the data into training and testing sets for each predictor
    xTrain, xTest, yTrain, yTest = train_test_split(predictors, y, test_size=0.3, random_state=42)
    
    # Initialize and train the decision tree classifier`
    tree = DecisionTreeClassifier(random_state=42,class_weight='balanced')
    scaler = StandardScaler()
    xTrainScaled = scaler.fit_transform(xTrain)
    xTestScaled = scaler.transform(xTest)
    
    tree.fit(xTrainScaled, yTrain)
    yProb = tree.predict_proba(xTestScaled)[:, 1]
    yPred = tree.predict(xTestScaled)
    roc_score = roc_auc_score(yTest, yProb)
    if roc_score < minroc:
        minroc = roc_score
        minCol = col
        false_positive_rate, true_positive_rate, threshold = roc_curve(yTest, yProb)
    
    print(f"Classification Report for {col}:\n", classification_report(yTest, yPred))
    print(f"ROC AUC for full Decision Tree model without {col}:", roc_score)
    print("---------------------------------------------------------------------------")


# In[18]:


print(f"The removal of {minCol} from the full model decreased the ROC AUC score to {minroc}, the most \ncompared to the removal of other predictors.")

plt.subplots(1, figsize=(10,10))
plt.title(f'ROC Curve for Decision Tree (Full Model without {minCol})')
plt.plot(false_positive_rate, true_positive_rate)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# ## Q4

# In[19]:


# Split the data into training and testing sets for each predictor
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.3, random_state=42)

forest = RandomForestClassifier(n_estimators=100, random_state=42, criterion='gini', class_weight='balanced')

scaler = StandardScaler()
xTrainScaled = scaler.fit_transform(xTrain)
xTestScaled = scaler.transform(xTest)

forest.fit(xTrainScaled, yTrain)

yProb = forest.predict_proba(xTestScaled)[:, 1]
yPred = forest.predict(xTestScaled)

# Retrieve the feature importances
importances = forest.feature_importances_

# Print the feature importances
print("Feature Importances:")
for pred, importance in zip(X.columns, importances):
    print(f"{pred} (Importance): {importance:.4f}")

print(classification_report(yTest, yPred))  

print("ROC AUC for full random forest model:", roc_auc_score(yTest, yProb))
matr = confusion_matrix(yTest, yPred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = matr, display_labels = [False, True])
cm_display.plot()
plt.show() 


# In[20]:


false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(yTest, yProb)
plt.subplots(1, figsize=(10,10))
plt.title('ROC Curve for Random Forest (Full Model)')
plt.plot(false_positive_rate1, true_positive_rate1)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[21]:


minCol = ''
minroc = 100

for col in X:
    predictors = X.drop([col],axis=1)    
    
    # Initialize and train the rf classifier`
    forest = RandomForestClassifier(n_estimators=50, random_state=42, criterion='gini', class_weight='balanced', n_jobs=-1)
    
    xTrain, xTest, yTrain, yTest = train_test_split(predictors, y, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    xTrainScaled = scaler.fit_transform(xTrain)
    xTestScaled = scaler.transform(xTest)
    
    forest.fit(xTrainScaled, yTrain)
    yProb = forest.predict_proba(xTestScaled)[:, 1]
    yPred = forest.predict(xTestScaled)
    roc_score = roc_auc_score(yTest, yProb)
    if roc_score < minroc:
        minroc = roc_score
        minCol = col
        false_positive_rate, true_positive_rate, threshold = roc_curve(yTest, yProb)    
    
    print(f"Classification Report for model without {col}:\n", classification_report(yTest, yPred))
    print(f"ROC AUC for full model without {col}:", roc_score)
    print("---------------------------------------------------------------------------")


# In[22]:


print(f"The removal of {minCol} from the full model decreased the ROC AUC score to {minroc}, the most \ncompared to the removal of other predictors.")

plt.subplots(1, figsize=(10,10))
plt.title(f'ROC Curve for Random Forest (Full Model without {minCol})')
plt.plot(false_positive_rate, true_positive_rate)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# ## Q5

# In[23]:


# Split the data into training and testing sets for each predictor
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.3, random_state=42)

bdt = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=2, class_weight='balanced'), algorithm="SAMME.R", n_estimators=100, learning_rate=1
)

scaler = StandardScaler()
xTrainScaled = scaler.fit_transform(xTrain)
xTestScaled = scaler.transform(xTest)

bdt.fit(xTrainScaled, yTrain)

yProb = bdt.predict_proba(xTestScaled)[:, 1]
yPred = bdt.predict(xTestScaled)

# Retrieve the feature importances
importances = bdt.feature_importances_

# Print the feature importances
print("Feature Importances:")
for pred, importance in zip(X.columns, importances):
    print(f"{pred} (Importance): {importance:.4f}")

print(classification_report(yTest, yPred))  

print("ROC AUC for full adaBoost model:", roc_auc_score(yTest, yProb))
matr = confusion_matrix(yTest, yPred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = matr, display_labels = [False, True])
cm_display.plot()
plt.show() 


# In[24]:


false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(yTest, yProb)
plt.subplots(1, figsize=(10,10))
plt.title('ROC Curve for adaBoost (Full Model)')
plt.plot(false_positive_rate1, true_positive_rate1)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[25]:


minCol = ''
minroc = 100

for col in X:
    predictors = X.drop([col],axis=1)

    # Split the data into training and testing sets for each predictor
    xTrain, xTest, yTrain, yTest = train_test_split(predictors, y, test_size=0.3, random_state=42)
    
    # Initialize and train the rf classifier`
    bdt = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=2, class_weight='balanced'), algorithm="SAMME.R", n_estimators=50, learning_rate=1
    )
    
    scaler = StandardScaler()
    xTrainScaled = scaler.fit_transform(xTrain)
    xTestScaled = scaler.transform(xTest)
    
    bdt.fit(xTrainScaled, yTrain)
    
    yPred = bdt.predict(xTestScaled)
    yProb = bdt.predict_proba(xTestScaled)[:, 1]
    roc_score = roc_auc_score(yTest, yProb)
    if roc_score < minroc:
        minroc = roc_score
        minCol = col
        false_positive_rate, true_positive_rate, threshold = roc_curve(yTest, yProb)   
    
    print(f"Classification Report for adaBoost model without {col}:\n", classification_report(yTest, yPred))
    print(f"ROC AUC for full adaBoost model without {col}:", roc_score)
    print("---------------------------------------------------------------------------")


# In[26]:


print(f"The removal of {minCol} from the full model decreased the ROC AUC score to {minroc}, the most \ncompared to the removal of other predictors.")

plt.subplots(1, figsize=(10,10))
plt.title(f'ROC Curve for adaBoost (Full Model without {minCol})')
plt.plot(false_positive_rate, true_positive_rate)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:




