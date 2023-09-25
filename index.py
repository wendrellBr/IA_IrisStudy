from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Carregando o conjunto de dados Iris
iris = datasets.load_iris()

# Montando o DataFrame com as características e o alvo
df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Exibindo o DataFrame
print(df)

# Dividindo os dados em características (X) e alvo (y)
X = df.drop('species', axis=1)
y = df['species']

# Dividindo os dados em conjuntos de treinamento e teste (80% treinamento, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizando as características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Exibindo as dimensões dos conjuntos de treinamento e teste
print("-"*50)
print("Dimensões dos conjuntos de treinamento e teste: ")
print("Conjunto de Treinamento (X):   ", X_train.shape)
print("Conjunto de Teste (X):         ", X_test.shape)
print("Conjunto de Treinamento (y):   ", y_train.shape)
print("Conjunto de Teste (y):         ", y_test.shape)

# Criar um modelo de Regressão Logística
model = LogisticRegression()

# Treinar o modelo com os dados de treinamento
model.fit(X_train, y_train)

# Fazer previsões com o modelo nos dados de teste
y_pred = model.predict(X_test)

# Calcular a precisão
accuracy = accuracy_score(y_test, y_pred)

# Exibir a matriz de confusão
confusion = confusion_matrix(y_test, y_pred)

# Exibir o relatório de classificação
report = classification_report(y_test, y_pred)

print("\nResultados da Avaliação do Modelo\n")
print("-"*50)
print(f"Precisão: {accuracy:.2f}")

print("-"*50)
print("Matriz de Confusão:")
print(confusion)

print("-"*50)
print("Relatório de Classificação:\n")
print(report)
print("-"*50)