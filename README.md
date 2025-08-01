
# Jawaban Ujian Data Titanic - Google Colab
# ------------------------------------------
# Dataset: https://bit.ly/datasettitanic

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# 1. Eksplorasi Awal
df = pd.read_csv('/content/Titanic-Dataset.csv')  # upload file manual di Colab

# a. Jumlah total penumpang
print("Total penumpang:", len(df))

# b. Jumlah penumpang selamat dan tidak selamat
print("Jumlah yang selamat dan tidak selamat:")
print(df['Survived'].value_counts())
sns.countplot(data=df, x='Survived')
plt.title('Survived vs Not Survived')
plt.xticks([0,1], ['Tidak Selamat','Selamat'])
plt.show()

# c. Rata-rata umur, termuda, tertua
print("Rata-rata umur:", df['Age'].mean())
print("Umur termuda:", df['Age'].min())
print("Umur tertua:", df['Age'].max())

# d. Jumlah berdasarkan jenis kelamin
print("Jumlah berdasarkan jenis kelamin:")
print(df['Sex'].value_counts())
sns.countplot(data=df, x='Sex')
plt.title('Jumlah berdasarkan Jenis Kelamin')
plt.show()

# e. Berdasarkan kelas
print("Jumlah berdasarkan kelas:")
print(df['Pclass'].value_counts())
sns.countplot(data=df, x='Pclass')
plt.title('Jumlah berdasarkan Kelas')
plt.show()
print("Persentase kelas 1:", (df['Pclass'].value_counts(normalize=True)[1]*100), "%")

# 2. Preprocessing
# a. Missing value
print("Missing value:")
print(df.isnull().sum())

# b. Isi Age dengan median, Embarked dengan modus
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin'], inplace=True)

# c. Deteksi outlier
def detect_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    return series[(series < Q1 - 1.5 * IQR) | (series > Q3 + 1.5 * IQR)]

print("Outlier Age:", len(detect_outliers(df['Age'])))
print("Outlier Fare:", len(detect_outliers(df['Fare'])))

# d. Scaling dan encoding
df_model = df.drop(columns=['PassengerId', 'Name', 'Ticket'])
le = LabelEncoder()
df_model['Sex'] = le.fit_transform(df_model['Sex'])
df_model['Embarked'] = le.fit_transform(df_model['Embarked'])

X = df_model.drop('Survived', axis=1)
y = df_model['Survived']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Modelling
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n{name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# 4. Evaluasi model terbaik
best_model = RandomForestClassifier()
best_model.fit(X_train, y_train)
y_pred_best = best_model.predict(X_test)
print("\nEvaluasi Model Terbaik (Random Forest):")
print("Classification Report:")
print(classification_report(y_test, y_pred_best))
print("Diprediksi Selamat:", sum(y_pred_best == 1))
print("Benar-benar Selamat:", sum((y_pred_best == 1) & (y_test == 1)))
