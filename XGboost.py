# Importar bibliotecas necess·rias
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Importa os dados iniciais
entrada = pd.read_excel("Entrada_simples.xlsx", sheet_name="entrada") # Base 1 (mercado 1)
entrada = pd.read_excel("Entrada_simples_v2.xlsx", sheet_name="entrada") # Base 1 (mercado 1 s/ tarifa e hor·rio)
entrada = pd.read_excel("Entrada_simples_v2.xlsx", sheet_name="entrada2") # Base 2 (mercado 1 + 2)
entrada = pd.read_excel("Entrada_simples_v2.xlsx", sheet_name="entrada3") # Base 3 (mercado 2)


# Remover todas as linhas com valores NA
dado_limpo = entrada.dropna()
dado_limpo = pd.read_csv('dado_limpo.csv')

# Dicion·rio com a classificaÁ„o que vocÍ deseja
category_mapping = {
    "Y": 0,
    "B": 1,
    "H": 2,
    "K": 3,
    "M": 4,
    "L": 5,
    "V": 6,
    "X": 7,
    "S": 8,
    "N": 9,
    "Q": 10,
    "O": 11,
    "G": 12,
    "TC": 99
}

# Colunas que precisam seguir essa classificaÁ„o
categorical_cols = ["CM0", "CM1", "CM2", "CM3", "match_min", "CL"]

# Aplicar o mapeamento nas colunas especificadas
for col in categorical_cols:
    dado_limpo[col] = dado_limpo[col].map(category_mapping)
    
# Dicion·rios com a classificaÁ„o para as novas colunas
season_mapping = {
    "HIGH": 1,
    "LOW": 2,
    "CARNAVAL": 3
}

clstr_mapping = {
    "MF": 4,
    "FO": 3,
    "MO": 2,
    "FR": 1
}

fluxo_mapping = {
    "Fluxo": 2,
    "CF": 1,
    "indefinido": 0
}

# Aplicar o mapeamento nas colunas especificadas
dado_limpo["Season"] = dado_limpo["Season"].map(season_mapping)
dado_limpo["CLSTR"] = dado_limpo["CLSTR"].map(clstr_mapping)
dado_limpo["Fluxo"] = dado_limpo["Fluxo"].map(fluxo_mapping)    
    



dado_limpo.to_csv('dado_limpo.csv', index=False)







    
#################################################################################### #
    
# Separar features e vari√°vel alvo
# X = dado_limpo.drop(columns=["CM0", "ORDEM", "TIME", "TIME.1"])  # Features
X = dado_limpo.drop(columns=["CM0"])  # Features
y = dado_limpo["CM0"]  # Vari√°vel alvo

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123, stratify=y)


# Hiperpar‚metros Modelo completo XGBoost
modelo_xgb = xgb.XGBClassifier(
    n_estimators=1000, 
    random_state=123,  
    eval_metric='mlogloss', 
    learning_rate=0.01, 
    max_depth=8, 
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=1.5
)


modelo_xgb.fit(X_train, y_train)

# Gerar previs√µes
previsoes = modelo_xgb.predict(X_test)

####################################################################################
# K-Fold Test
 from sklearn.model_selection import StratifiedKFold, cross_val_score
 import numpy as np

 # Definir o StratifiedKFold cross-validator
 skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

 # Avaliar o modelo com K-fold Cross Validation usando estratificaÁ„o
 scores = cross_val_score(modelo_rf, X, y, cv=skf, scoring='accuracy')

 # Exibir as pontuaÁıes e a mÈdia
 print("Scores em cada fold: ", scores)
 print("Acur·cia mÈdia: ", np.mean(scores))

from sklearn.metrics import cohen_kappa_score

# Calcular o Cohen's Kappa
kappa = cohen_kappa_score(y_test, previsoes)

print(f"Cohen's Kappa: {kappa:.4f}")

####################################################################################
# AVALIA√á√ÉO DO MODELO #
# MATRIZ DE CONFUS√ÉO #

# Definir as classes de 0 a 12
classes = list(range(13))
classes_originais = ['Y', 'B', 'H', 'K', 'M', 'L', 'V', 'X', 'S', 'N', 'Q', 'O', 'G']


# Gerar a matriz de confus√£o
matriz_confusao = confusion_matrix(y_test, previsoes, labels=classes)

# Verificar a matriz de confus√£o para garantir que esteja correta
print(matriz_confusao)

# Criar DataFrame a partir da matriz de confus√£o para plottar o gr√°fico
conf_df = pd.DataFrame(matriz_confusao, index=classes, columns=classes)


# Plotar a matriz de confus„o
plt.figure(figsize=(8, 6))
sns.heatmap(matriz_confusao, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confus„o - XGboost')
plt.xlabel('Previsto')
plt.ylabel('Observado')

plt.savefig("grafico_confusion.pdf", format='pdf')
plt.show()
####################################################################################
####################################################################################

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import numpy as np
# Avalia√ß√£o do modelo

# Acur√°cia
acuracia = accuracy_score(y_test, previsoes)
print(f"Acur√°cia: {acuracia:.4f}")

# Contagem de inst‚ncias por classe
from collections import Counter
pesos = np.array(list(Counter(y_test).values()))

# Precis√£o, Sensibilidade (Recall) e F1-Score para cada classe
precisao = precision_score(y_test, previsoes, average=None)
sensibilidade = recall_score(y_test, previsoes, average=None)
f1_scores = f1_score(y_test, previsoes, average=None)

# Exibir as m√©tricas para cada classe
for i, classe in enumerate(classes):
    print(f"\nClasse: {classe}")
    print(f"Precis√£o: {precisao[i]:.4f}")
    print(f"Sensibilidade (Recall): {sensibilidade[i]:.4f}")
    print(f"F1-Score: {f1_scores[i]:.4f}")
    
# C·lculo da mÈdia ponderada para Precis„o, Sensibilidade e F1-Score
precisao_ponderada = np.average(precisao, weights=pesos)
sensibilidade_ponderada = np.average(sensibilidade, weights=pesos)
f1_score_ponderado = np.average(f1_scores, weights=pesos)    
    
print("\nMÈdias Ponderadas:")
print(f"Precis„o Ponderada: {precisao_ponderada:.4f}")
print(f"Sensibilidade Ponderada: {sensibilidade_ponderada:.4f}")
print(f"F1-Score Ponderado: {f1_score_ponderado:.4f}")    
    

# Relat√≥rio de classifica√ß√£o completo
print("\nRelat√≥rio de Classifica√ß√£o Completo:\n")
print(classification_report(y_test, previsoes, target_names=[str(c) for c in classes]))

# Para visualizar a precis√£o, recall e F1-score de maneira tabular
resultados_parametricos = pd.DataFrame({
    'Classe': classes,
    'Precisao': precisao,
    'Sensibilidade (Recall)': sensibilidade,
    'F1-Score': f1_scores
})

print("\nTabela Resumida de Resultados:\n")
print(resultados_parametricos)

####################################################################################
####################################################################################
# TESTE CGR

# Importa os dados iniciais CGR
entrada_CGR = pd.read_excel("Entrada_simples_v2.xlsx", sheet_name="entrada4")

# Remover todas as linhas com valores NA
dado_limpoCGR = entrada_CGR.dropna()

# Nova base de teste (ou nova base)
for col in categorical_cols:
    dado_limpoCGR[col] = dado_limpoCGR[col].map(category_mapping) # Transformar os dados de teste com o encoder do treino


# Aplicar o mapeamento nas colunas especificadas
dado_limpoCGR["Season"] = dado_limpoCGR["Season"].map(season_mapping)
dado_limpoCGR["CLSTR"] = dado_limpoCGR["CLSTR"].map(clstr_mapping)
dado_limpoCGR["Fluxo"] = dado_limpoCGR["Fluxo"].map(fluxo_mapping)   


X_CGR = dado_limpoCGR.drop(columns=["CM0"])  # Features
y_CGR = dado_limpoCGR["CM0"]  # Vari√°vel alvo

# Alinhar colunas entre o modelo treinado (X_train) e a nova base (X_CGR)
X_CGR = X_CGR.reindex(columns=X_train.columns, fill_value=0)


# Gerar previs√µes CGR
previsoes_CGR = modelo_xgb.predict(X_CGR)

# Gerar a matriz de confus√£o
matriz_confusaoCGR = confusion_matrix(y_CGR, previsoes_CGR, labels=classes)

# Verificar a matriz de confus√£o para garantir que esteja correta
print(matriz_confusaoCGR)

# Acur√°cia
acuraciaCGR = accuracy_score(y_CGR, previsoes_CGR)
print(f"Acur√°cia: {acuraciaCGR:.4f}")

# Criar DataFrame a partir da matriz de confus√£o para plottar o gr√°fico
conf_dfCGR = pd.DataFrame(matriz_confusaoCGR, index=classes, columns=classes)

# Plotar o heatmap com cores e anota√ß√µes ajustadas
plt.figure(figsize=(10, 8))
sns.heatmap(conf_dfCGR, annot=True, fmt='d', cmap='Blues', cbar=False, 
            annot_kws={"color": "black"})  # Cor das anota√ß√µes

import numpy as np  # Importar NumPy


# Inicializar uma vari√°vel para a acur√°cia ponderada
weighted_accuracy = 0
total_instances = np.sum(matriz_confusaoCGR)

for i in range(len(matriz_confusaoCGR)):
    for j in range(len(matriz_confusaoCGR)):
        # Ponderar o acerto completo com peso 1
        if i == j:
            weighted_accuracy += matriz_confusaoCGR[i, j]
        # Ponderar erros adjacentes com peso 0.5
        elif abs(i - j) == 1:
            weighted_accuracy += 0.5 * matriz_confusaoCGR[i, j]

weighted_accuracy /= total_instances

print(f"Acur√°cia ponderada: {weighted_accuracy:.4f}")

###########################################################################
###########################################################################
# ACUR√ÅCIA POR AGRUPAMENTO CGR

import numpy as np
from sklearn.metrics import accuracy_score

# Dicion√°rio de mapeamento reverso (inteiros para classes)
mapeamento_reverso = {
    0: 'Y', 1: 'B', 2: 'H', 3: 'K', 4: 'M', 5: 'L', 6: 'V', 
    7: 'X', 8: 'S', 9: 'N', 10: 'Q', 11: 'O', 12: 'G'
}

# Fun√ß√£o para aplicar o mapeamento reverso
def reverter_para_classes(numeros):
    return [mapeamento_reverso[numero] for numero in numeros]

# Suponha que 'y_test' e 'previsoes' sejam arrays de n√∫meros inteiros (0, 1, 2, etc.)
# Reverter os inteiros para as classes originais
y_test_classes_CGR = reverter_para_classes(y_CGR)  # Classes reais
previsoes_classes_CGR = reverter_para_classes(previsoes_CGR)  # Previs√µes

# Definir os grupos de mix
mix_high = ['Y', 'B', 'H', 'K']
mix_mid = ['M', 'L', 'V', 'X', 'S', 'N']
mix_low = ['Q', 'O', 'G']

# Fun√ß√£o para mapear as classes para seus respectivos grupos
def map_to_mix_group(label):
    if label in mix_high:
        return 'MIX HIGH'
    elif label in mix_mid:
        return 'MIX MID'
    elif label in mix_low:
        return 'MIX LOW'
    else:
        return 'UNKNOWN'  # Caso tenha algum valor inesperado

# Agora, aplicar o mapeamento para os grupos de mix nas classes reais e previstas
y_test_mix_group_cgr = [map_to_mix_group(label) for label in y_test_classes_CGR]
previsoes_mix_group_cgr = [map_to_mix_group(label) for label in previsoes_classes_CGR]

# Calcular a acur√°cia total por grupo de mix
acuracia_mix_cgr = accuracy_score(y_test_mix_group_cgr, previsoes_mix_group_cgr)
print(f"Acur√°cia por grupo de mix: {acuracia_mix:.4f}")

# Calcular a acur√°cia separada para cada grupo (MIX HIGH, MIX MID, MIX LOW)
for mix_group in ['MIX HIGH', 'MIX MID', 'MIX LOW']:
    y_test_group_cgr = [1 if label == mix_group else 0 for label in y_test_mix_group_cgr]
    previsoes_group_cgr = [1 if label == mix_group else 0 for label in previsoes_mix_group_cgr]
    
    acuracia_grupo = accuracy_score(y_test_group_cgr, previsoes_group_cgr)
    print(f"Acur√°cia para {mix_group}: {acuracia_grupo:.4f}")
    
 #############
# TESTE Comum   agrupamento 
    
 import numpy as np
 from sklearn.metrics import accuracy_score

 # Dicion√°rio de mapeamento reverso (inteiros para classes)
 mapeamento_reverso = {
     0: 'Y', 1: 'B', 2: 'H', 3: 'K', 4: 'M', 5: 'L', 6: 'V', 
     7: 'X', 8: 'S', 9: 'N', 10: 'Q', 11: 'O', 12: 'G'
 }

 # Fun√ß√£o para aplicar o mapeamento reverso
 def reverter_para_classes(numeros):
     return [mapeamento_reverso[numero] for numero in numeros]

 # Suponha que 'y_test' e 'previsoes' sejam arrays de n√∫meros inteiros (0, 1, 2, etc.)
 # Reverter os inteiros para as classes originais
 y_test_classes = reverter_para_classes(y_test)  # Classes reais
 previsoes_classes = reverter_para_classes(previsoes)  # Previs√µes

 # Definir os grupos de mix
 mix_high = ['Y', 'B', 'H', 'K']
 mix_mid = ['M', 'L', 'V', 'X', 'S', 'N']
 mix_low = ['Q', 'O', 'G']

 # Fun√ß√£o para mapear as classes para seus respectivos grupos
 def map_to_mix_group(label):
     if label in mix_high:
         return 'MIX HIGH'
     elif label in mix_mid:
         return 'MIX MID'
     elif label in mix_low:
         return 'MIX LOW'
     else:
         return 'UNKNOWN'  # Caso tenha algum valor inesperado

 # Agora, aplicar o mapeamento para os grupos de mix nas classes reais e previstas
 y_test_mix_group = [map_to_mix_group(label) for label in y_test_classes]
 previsoes_mix_group = [map_to_mix_group(label) for label in previsoes_classes]

 # Calcular a acur√°cia total por grupo de mix
 acuracia_mix = accuracy_score(y_test_mix_group, previsoes_mix_group)
 print(f"Acur√°cia por grupo de mix: {acuracia_mix:.4f}")

 # Calcular a acur√°cia separada para cada grupo (MIX HIGH, MIX MID, MIX LOW)
 for mix_group in ['MIX HIGH', 'MIX MID', 'MIX LOW']:
     y_test_group = [1 if label == mix_group else 0 for label in y_test_mix_group]
     previsoes_group = [1 if label == mix_group else 0 for label in previsoes_mix_group]
     
     acuracia_grupo = accuracy_score(y_test_group, previsoes_group)
     print(f"Acur√°cia para {mix_group}: {acuracia_grupo:.4f}")   

###########################################################################
###########################################################################
# Exportar para um arquivo Excel

previsoes_finalCGR = pd.DataFrame(previsoes_CGR, columns=['Previsoes'])

previsao = pd.DataFrame(previsoes, columns = ['Previs√£o'])
X_train.to_excel("X_train_export.xlsx", index=False)  # O index=False remove o √≠ndice do arquivo
X_test.to_excel("X_test_export.xlsx", index=False)  # O index=False remove o √≠ndice do arquivo
y_test.to_excel("y_test_exportXGBoost.xlsx", index=False)  # O index=False remove o √≠ndice do arquivo
previsoes_series.to_excel("previsoes_testexportXGBoost.xlsx", index=False)  # O index=False remove o √≠ndice do arquivo
dado_limpo.to_excel("baseinteira_test_export.xlsx", index=False)  # O index=False remove o √≠ndice do arquivo

previsoes_series = pd.Series(previsoes)

##############


# Exemplo de treino no XGBoost com dados simulados
dtrain = xgb.DMatrix(X_train, label=y_train)
params = {
    'objective': 'multi:softprob',  # Multi-classe
    'num_class': len(set(y_train)),  # N˙mero de classes
    'max_depth': 4
}
model = xgb.train(params, dtrain, num_boost_round=10)

# Visualizar uma ·rvore especÌfica (exemplo: a primeira ·rvore)
xgb.plot_tree(model, num_trees=0)
plt.show()

#####################



# Exemplo de treino no XGBoost com dados simulados
dtrain = xgb.DMatrix(X_train, label=y_train)
params = {
    'objective': 'multi:softprob',  # Multi-classe
    'num_class': len(set(y_train)),  # N˙mero de classes
    'max_depth': 4
}
model = xgb.train(params, dtrain, num_boost_round=10)

# Exportar a ·rvore de decis„o no formato dot para o Graphviz
dot_data = xgb.to_graphviz(model, num_trees=0)  # Escolha qual ·rvore exportar


# Renderizar e visualizar o arquivo diretamente com graphviz
dot_data.render("xgboost_tree", format="pdf")  # Salvar como arquivo PDF
dot_data.view()  # Abrir o PDF gerado automaticamente



import pandas as pd

# Carregar o CSV como DataFrame
dado_limpo = pd.read_csv('dado_limpo.csv')

# Verificar quantas observaÁıes tem cada classe
contagem_classes = dado_limpo['CM0'].value_counts()
print("DistribuiÁ„o original das classes:\n", contagem_classes)

# Definir o n˙mero mÌnimo de observaÁıes (mÌnimo da classe menos representada)
n_min = contagem_classes.min()

# Criar um DataFrame vazio para armazenar as amostras balanceadas
dado_balanceado = pd.DataFrame()

# Fazer undersampling: selecionar amostras aleatÛrias de cada classe para igualar o tamanho da menor classe
for classe in contagem_classes.index:
    amostras_classe = dado_limpo[dado_limpo['CM0'] == classe].sample(n=n_min, random_state=123)
    dado_balanceado = pd.concat([dado_balanceado, amostras_classe])


# Verificar a nova distribuiÁ„o das classes
print("DistribuiÁ„o balanceada das classes:\n", dado_balanceado['CM0'].value_counts())
