import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Configurando o estilo dos gráficos
sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")

# Carregando os dados do arquivo "heart.csv"
df = pd.read_csv("heart.csv")
print("Dados iniciais:")
print(df.head())
print("---------------------------------")

# Exibindo estatísticas descritivas do dataset
pd.set_option("display.float", "{:.2f}".format)
print("Estatísticas Descritivas:")
print(df.describe())
print("--------------------------------")

# Gráfico de barras para a variável alvo (target)
fig_barplot = plt.figure(num="Gráfico de Barras - Variável Alvo")
df.target.value_counts().plot(kind="bar", color=["salmon", "lightblue"])
plt.xlabel("Doença Cardíaca")
plt.ylabel("Contagem")
plt.title("Contagem de Doenças Cardíacas")
plt.show()

# Verificando valores ausentes
print("Valores ausentes:")
print(df.isna().sum())
print("-------------------------------------------")

categorical_val = []
continuous_val = []

# Identificando variáveis categóricas e contínuas
for column in df.columns:
    print('==============================')
    print(f"{column} : {df[column].unique()}")
    if len(df[column].unique()) <= 10:
        categorical_val.append(column)
    else:
        continuous_val.append(column)

print("-----------------------------------------------------")

# Gráficos de histograma para variáveis categóricas
plt.figure(figsize=(15, 15))
for i, column in enumerate(categorical_val, 1):
    plt.subplot(3, 3, i)
    df[df["target"] == 0][column].hist(bins=35, color='blue', label='Doença Cardíaca = NÃO', alpha=0.6)
    df[df["target"] == 1][column].hist(bins=35, color='red', label='Doença Cardíaca = SIM', alpha=0.6)
    plt.legend()
    plt.xlabel(column)
    plt.title("Histograma - " + column)
plt.show()

print("-----------------------------------------------------")

# Gráficos de histograma para variáveis contínuas
plt.figure(figsize=(15, 15))
for i, column in enumerate(continuous_val, 1):
    plt.subplot(3, 2, i)
    df[df["target"] == 0][column].hist(bins=35, color='blue', label='Doença Cardíaca = NÃO', alpha=0.6)
    df[df["target"] == 1][column].hist(bins=35, color='red', label='Doença Cardíaca = SIM', alpha=0.6)
    plt.legend()
    plt.xlabel(column)
    plt.title("Histograma - " + column)
plt.show()

print("-----------------------------------------------------")

# Gráfico de dispersão da idade em relação à frequência cardíaca máxima
plt.figure(figsize=(10, 8))
plt.scatter(df.age[df.target==1], df.thalachh[df.target==1], c="salmon")
plt.scatter(df.age[df.target==0], df.thalachh[df.target==0], c="lightblue")
plt.title("Doença Cardíaca em Função da Idade e Frequência Cardíaca Máxima")
plt.xlabel("Idade")
plt.ylabel("Frequência Cardíaca Máxima")
plt.legend(["Doença Cardíaca", "Sem Doença Cardíaca"])
plt.show()

print("-----------------------------------------------------")

# Matriz de correlação
corr_matrix = df.corr()
fig_corr = plt.figure(figsize=(15, 15))
ax = sns.heatmap(corr_matrix, annot=True, linewidths=0.5, fmt=".2f", cmap="YlGnBu")
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.title("Matriz de Correlação")
plt.show()

print("-----------------------------------------------------")

# Correlação com a variável alvo (target)
corr_with_target = df.drop('target', axis=1).corrwith(df.target)
corr_with_target.plot(kind='bar', grid=True, figsize=(12, 8), title="Correlação com a Variável Alvo")
plt.show()

print("-----------------------------------------------------")

# Pré-processamento e escalonamento dos dados
categorical_val.remove('target')
dataset = pd.get_dummies(df, columns=categorical_val)

scaler = StandardScaler()
col_to_scale = ['age', 'trstbps', 'chol', 'thalachh', 'oldpeak']
dataset[col_to_scale] = scaler.fit_transform(dataset[col_to_scale])

print("-----------------------------------------------------")

# Função para exibir resultados da classificação
def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
        print("Resultado do Treinamento:\n================================================")
        print(f"Acurácia: {accuracy_score(y_train, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"RELATÓRIO DE CLASSIFICAÇÃO:\n{clf_report}")
        print("_______________________________________________")
        print(f"Matriz de Confusão:\n{confusion_matrix(y_train, pred)}\n")
    elif not train:
        pred = clf.predict(X_test)
        clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        print("Resultado do Teste:\n================================================")
        print(f"Acurácia: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"RELATÓRIO DE CLASSIFICAÇÃO:\n{clf_report}")
        print("_______________________________________________")
        print(f"Matriz de Confusão:\n{confusion_matrix(y_test, pred)}\n")

# Divisão dos dados em treinamento e teste
X = dataset.drop('target', axis=1)
y = dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Criação e treinamento do modelo de Regressão Logística
lr_clf = LogisticRegression(solver='liblinear')
lr_clf.fit(X_train, y_train)

# Avaliação do modelo
print_score(lr_clf, X_train, y_train, X_test, y_test, train=True)
print_score(lr_clf, X_train, y_train, X_test, y_test, train=False)

print("-----------------------------------------------------")

# Resultados da Regressão Logística
test_score = accuracy_score(y_test, lr_clf.predict(X_test)) * 100
train_score = accuracy_score(y_train, lr_clf.predict(X_train)) * 100

results_df = pd.DataFrame(data=[["Regressão Logística", train_score, test_score]],
                          columns=['Modelo', 'Acurácia Treinamento %', 'Acurácia Teste %'])
print("Resultados da Regressão Logística:")
print(results_df)
