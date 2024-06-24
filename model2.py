import pandas as pd
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df=pd.read_csv('Debernardi et al 2020 data.csv')
unique_values = []
for value in df['sample_origin']:
    if value not in unique_values:
        unique_values.append(value)

df.drop(['sample_id'], axis=1, inplace=True)
sns.countplot(x='diagnosis', hue='sex', data=df)
def gender(sex):
    if sex=='M':
        return 0
    elif sex=='F':
        return 1
    else:
        return -1

df['sex'] = df['sex'].apply(gender)
def new_age(age):
    if 25 <= age <= 35:
        return 0
    elif 35 <= age <= 50:
        return 1
    elif 50 <= age <= 90:
        return 2
    else:
        return -1

df['age'] = df['age'].apply(new_age)

sns.countplot(x='diagnosis', hue='age', data=df)
def cohort(patient_cohort):
    if patient_cohort=='Cohort1':
        return 0
    elif patient_cohort=='Cohort2':
        return 1
    else:
        return -1

df['patient_cohort'] = df['patient_cohort'].apply(cohort)
def origin(sample_origin):
    if sample_origin=='BPTB':
        return 0
    else:
        return 1

df['sample_origin'] = df['sample_origin'].apply(origin)

def cr(creatinine):
    if creatinine <= 0.37320:
        return 0
    elif 0.38 <= creatinine<= 1.139:
        return 1
    elif creatinine > 1.14:
        return 2
    else:
        return -1

df['creatinine'] = df['creatinine'].apply(cr)

df_tr=df.drop(['stage','benign_sample_diagnosis','plasma_CA19_9','REG1A','diagnosis'], axis=1)

X = df_tr
y = df['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.21, random_state=42)
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

from sklearn.ensemble import GradientBoostingClassifier

gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train_scaled, y_train)

y_pred = gb_model.predict(X_test_scaled)
print(y_pred)
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)


from xgboost import XGBClassifier

xgb_model = XGBClassifier(random_state=42)
xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)
