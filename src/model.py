import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score
data = pd.read_csv("../data/diabetes.csv")
cols = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
for col in cols:
    data[col] = data[col].replace(0, np.nan)
    data[col] = data[col].fillna(data[col].median())
    x = data.drop('Outcome', axis=1)
y = data['Outcome']
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
model1 = LogisticRegression(class_weight='balanced')
model2 = SVC()
model3 = RandomForestClassifier()
model4 = GradientBoostingClassifier(n_estimators=1000)
def cal(model):
    model.fit(x_train, y_train)
    pre = model.predict(x_test)

    accuracy = accuracy_score(y_test, pre)
    recall = recall_score(y_test, pre)
    f1 = f1_score(y_test, pre)

    print(model)
    print("accuracy:", accuracy)
    print("recall:", recall)
    print("f1:", f1)

    cm = confusion_matrix(y_test, pre)

    df_cm = pd.DataFrame(cm,
                         index=["Actual 0", "Actual 1"],
                         columns=["Pred 0", "Pred 1"])

    print(df_cm)

    sns.heatmap(cm, annot=True)
    plt.show()
    probs = model1.predict_proba(x_test)[:,1]
    y_pred = (probs > 0.3).astype(int)
    return model

        
trained_model = cal(model1)


new_patient = pd.DataFrame(
    [[2,120,70,25,80,30.5,0.5,35]],
    columns=x.columns
)

new_patient = scaler.transform(new_patient)

prediction = trained_model.predict(new_patient)

print(prediction)
proba = model1.predict_proba(new_patient)
print(proba)
if proba[0][1] > 0.7:
    print("High risk")
elif proba[0][1] > 0.3:
    print("Medium risk")
else:
    print("Low risk")