# Importar as bibliotecas necess√°rias
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
#%%
# Importa os dados iniciais
entrada = pd.read_excel("Entrada_simples_v2.xlsx", sheet_name="entrada")
entrada = pd.read_excel("Entrada_simples_v2.xlsx", sheet_name="entrada2") # Base 2 (mercado 1 + 2)
entrada = pd.read_excel("Entrada_simples_v2.xlsx", sheet_name="entrada3") # Base 3 (mercado 2)
#%%
# Remover todas as linhas com valores NA
dado_limpo = entrada.dropna()

# Dicion√°rio com a classifica√ß√£o que voc√™ deseja
category_mapping = {
    "Y": 0, "B": 1, "H": 2, "K": 3, "M": 4, "L": 5, "V": 6, 
    "X": 7, "S": 8, "N": 9, "Q": 10, "O": 11, "G": 12, "TC":99
}
# Colunas que precisam seguir essa classifica√ß√£o
categorical_cols = ["CM0", "CM1", "CM2", "CM3", "match_min", "CL"]

# Aplicar o mapeamento nas colunas especificadas
for col in categorical_cols:
    dado_limpo[col] = dado_limpo[col].map(category_mapping)

# Dicion√°rios com a classifica√ß√£o para as novas colunas
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
# Separar features e vari√°vel alvo
X = dado_limpo.drop(columns=["CM0"])  # Features
y = dado_limpo["CM0"]  # Vari√°vel alvo

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
#%%
# Modelo de √°rvore de decis√£o simples
modelo_dt = DecisionTreeClassifier(random_state=123)
#%%
# ¡rvore de Decis„o complexa

modelo_dt = DecisionTreeClassifier(
    criterion='gini',
    splitter='best',
    max_depth=10,                 # Permitindo maior profundidade
    min_samples_split=4,          # Reduzido para permitir divisıes finas
    min_samples_leaf=1,           # Reduzido para permitir nÛs folha pequenos
    min_weight_fraction_leaf=0.0, # Nenhuma restriÁ„o no peso das folhas
    max_features=None,          # Considera log2 das features em cada split
    random_state=123,
    max_leaf_nodes=None,          # Sem limite de nÛs folha
    min_impurity_decrease=0.0,    # Divisıes menores s„o permitidas
    class_weight=None,      # Mantido para lidar com desbalanceamento
    ccp_alpha=0.0                 # Sem poda inicial, evitando subajuste
)
#%%
modelo_dt.fit(X_train, y_train)

# Gerar previs√µes
previsoes_dt = modelo_dt.predict(X_test)

# Avaliar o modelo
acuracia = accuracy_score(y_test, previsoes_dt)
print(f"Acur√°cia: {acuracia:.4f}")

# Matriz de confus√£o
matriz_confusao = confusion_matrix(y_test, previsoes_dt)

# Plotar a matriz de confus√£o
plt.figure(figsize=(8, 6))
sns.heatmap(matriz_confusao, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confus√£o - √Årvore de Decis√£o')
plt.xlabel('Previsto')
plt.ylabel('Observado')
plt.show()
#%%



############################################
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
previsoes_dt_CGR = modelo_dt.predict(X_CGR)

# Gerar a matriz de confus√£o
matriz_confusaoCGR = confusion_matrix(y_CGR, previsoes_dt_CGR, labels=classes)

# Verificar a matriz de confus√£o para garantir que esteja correta
print(matriz_confusaoCGR)

# Acur√°cia
acuraciaCGR = accuracy_score(y_CGR, previsoes_dt_CGR)
print(f"Acur√°cia: {acuraciaCGR:.4f}")

#%%
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

 # Suponha que 'y_test' e 'previsoes_dt' sejam arrays de n√∫meros inteiros (0, 1, 2, etc.)
 # Reverter os inteiros para as classes originais
 y_test_classes = reverter_para_classes(y_test)  # Classes reais
 previsoes_dt_classes = reverter_para_classes(previsoes_dt)  # Previs√µes

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
 previsoes_dt_mix_group = [map_to_mix_group(label) for label in previsoes_dt_classes]

 # Calcular a acur√°cia total por grupo de mix
 acuracia_mix = accuracy_score(y_test_mix_group, previsoes_dt_mix_group)
 print(f"Acur√°cia por grupo de mix: {acuracia_mix:.4f}")

 # Calcular a acur√°cia separada para cada grupo (MIX HIGH, MIX MID, MIX LOW)
 for mix_group in ['MIX HIGH', 'MIX MID', 'MIX LOW']:
     y_test_group = [1 if label == mix_group else 0 for label in y_test_mix_group]
     previsoes_dt_group = [1 if label == mix_group else 0 for label in previsoes_dt_mix_group]
     
     acuracia_grupo = accuracy_score(y_test_group, previsoes_dt_group)
     print(f"Acur√°cia para {mix_group}: {acuracia_grupo:.4f}")   
    
 ######################################
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import numpy as np
# Avalia√ß√£o do modelo
# Definir as classes de 0 a 12
classes = list(range(13))
classes_originais = ['Y', 'B', 'H', 'K', 'M', 'L', 'V', 'X', 'S', 'N', 'Q', 'O', 'G']
# Acur√°cia
acuracia = accuracy_score(y_test, previsoes_dt)
print(f"Acur√°cia: {acuracia:.4f}")

# Contagem de inst‚ncias por classe
from collections import Counter
pesos = np.array(list(Counter(y_test).values()))

# Precis√£o, Sensibilidade (Recall) e F1-Score para cada classe
precisao = precision_score(y_test, previsoes_dt, average=None)
sensibilidade = recall_score(y_test, previsoes_dt, average=None)
f1_scores = f1_score(y_test, previsoes_dt, average=None)

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


#####################
#%%
X_test.to_excel("X_test_export.xlsx", index=False)  # O index=False remove o √≠ndice do arquivo
y_test.to_excel("y_test_exportXGBoost.xlsx", index=False)  # O index=False remove o √≠ndice do arquivo
previsoes_dt_series.to_excel("previsoes_dt_testexportXGBoost.xlsx", index=False)  # O index=False remove o √≠ndice do arquivo 

previsoes_dt_series = pd.Series(previsoes_dt) 

#%%

import sys
print(sys.executable)
#%%
### busca de melhores par‚metros
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# EspaÁo de busca para a ·rvore de decis„o
dt_params = {
    'criterion': ['gini', 'entropy', 'log_loss'],             # CritÈrio de divis„o
    'splitter': ['best', 'random'],                           # EstratÈgia de divis„o
    'max_depth': [None, 10, 25, 50],                          # Profundidade m·xima da ·rvore
    'min_samples_split': [2, 4, 8, 16],                       # MÌnimo de amostras para dividir um nÛ
    'min_samples_leaf': [1, 2, 4, 8],                         # MÌnimo de amostras em um nÛ folha
    'min_weight_fraction_leaf': [0.0, 0.01, 0.05],            # FraÁ„o mÌnima do peso em um nÛ folha
    'max_features': ['sqrt', 'log2', None],                   # N˙mero m·ximo de features para cada split
    'max_leaf_nodes': [None, 10, 50, 100],                    # N˙mero m·ximo de nÛs folha
    'min_impurity_decrease': [0.0, 0.001, 0.01],              # ReduÁ„o mÌnima de impureza para dividir um nÛ
    'class_weight': [None, 'balanced'],                       # Balanceamento das classes
    'ccp_alpha': [0.0, 0.001, 0.01, 0.1]                      # Par‚metro de poda de complexidade
}

# Para a ¡rvore de Decis„o
dt_model = DecisionTreeClassifier(random_state=123)
dt_grid_search = GridSearchCV(estimator=dt_model, param_grid=dt_params, cv=5, scoring='accuracy', n_jobs=-1)

# Executar a busca de hiperpar‚metros para a ¡rvore de Decis„o
dt_grid_search.fit(X_train, y_train)
print("Melhores par‚metros para a ¡rvore de Decis„o:", dt_grid_search.best_params_)
#%%
import numpy as np
from sklearn.metrics import top_k_accuracy_score

# Obter as probabilidades de previs„o para cada classe
y_pred_proba = modelo_dt.predict_proba(X_test)

# Definir o valor de K
K = 4

# Calcular o Top-K Accuracy
top_k_acc = top_k_accuracy_score(y_test, y_pred_proba, k=K)
print(f"Top-{K} Accuracy: {top_k_acc:.4f}")


# Ordenar as probabilidades para cada inst‚ncia de forma decrescente e selecionar as Top-K
sorted_probas = np.sort(y_pred_proba, axis=1)[:, -K:]
# Inverter a ordem para que as probabilidades estejam do mais prov·vel ao menos prov·vel
top_k_probas = sorted_probas[:, ::-1]

# Calcular a mediana das probabilidades para a 1™, 2™ e 3™ escolhas
median_proba_1st = np.median(top_k_probas[:, 0])
median_proba_2nd = np.median(top_k_probas[:, 1])
median_proba_3rd = np.median(top_k_probas[:, 2])

print(f"Mediana da probabilidade da 1™ escolha: {median_proba_1st:.4f}")
print(f"Mediana da probabilidade da 2™ escolha: {median_proba_2nd:.4f}")
print(f"Mediana da probabilidade da 3™ escolha: {median_proba_3rd:.4f}")
#%%
import numpy as np
import pandas as pd

# Definir o valor de K (Top-K)
K = 3

# Obter as probabilidades previstas para cada classe
y_pred_proba = modelo_dt.predict_proba(X_test)

# Ordenar as probabilidades em cada linha em ordem crescente e selecionar as Top-K
sorted_probas = np.sort(y_pred_proba, axis=1)[:, -K:]

# Inverter para obter as Top-K probabilidades em ordem decrescente (do mais prov·vel ao menos prov·vel)
top_k_probas = sorted_probas[:, ::-1]

# Criar um DataFrame para mostrar as primeiras 10 linhas com as 3 probabilidades mais altas para cada linha
top_k_df = pd.DataFrame(top_k_probas[:10], columns=[f"Top-{i+1} Probability" for i in range(K)])

# Exibir o DataFrame
print("Top-3 probabilidades para as primeiras 10 linhas:")
print(top_k_df)
#%%
import numpy as np

# Suponha que `y_test` e `previsoes_dt` sejam arrays contendo as classes verdadeiras e as classes previstas
# Certifique-se de que `y_test` e `previsoes_dt` s„o do tipo numpy array
y_test = np.array(y_test)  # Converte `y_test` para numpy array, se necess·rio
previsoes_dt = np.array(previsoes_dt)  # Converte `previsoes_dt` para numpy array, se necess·rio

# Filtrar Ìndices onde houve erro
erro_indices = np.where(y_test != previsoes_dt)[0]

# Calcular a diferenÁa absoluta entre a classe correta e a prevista para esses erros
erro_distancia_dt = np.abs(y_test[erro_indices] - previsoes_dt[erro_indices])

# Calcular a mÈdia e o desvio padr„o da dist‚ncia dos erros
mean_distancia_erro = np.mean(erro_distancia_dt)
std_distancia_erro = np.std(erro_distancia_dt)
qtd_erro = len(erro_indices)

# ConfiguraÁ„o das faixas (bins) para as dist‚ncias dos erros, limitando a 12 bins
bins = np.arange(1, 13, 1)

def calcular_frequencia_erro(erro_distancia_dt, bins):
    freq, _ = np.histogram(erro_distancia_dt, bins=bins)
    # Ajustar para 12 posiÁıes, adicionando zeros onde necess·rio
    freq_ajustado = np.zeros(12, dtype=int)
    freq_ajustado[:len(freq)] = freq
    return freq_ajustado

freq_dt = calcular_frequencia_erro(erro_distancia_dt, bins)

print(f"Quantidade de erros: {qtd_erro}")
print(f"MÈdia da dist‚ncia dos erros: {mean_distancia_erro:.4f}")
print(f"Desvio padr„o da dist‚ncia dos erros: {std_distancia_erro:.4f}")
print(freq_dt)

#%%
import pickle

with open('y_test.pkl', 'wb') as f:
    pickle.dump(y_test, f)

# Salvar `previsoes_dt`
with open('previsoes_dt.pkl', 'wb') as f:
    pickle.dump(previsoes_dt, f)
    #%%
    import numpy as np
    np.save('y_test.npy', y_test)
    np.save('previsoes_dt.npy', previsoes_dt)