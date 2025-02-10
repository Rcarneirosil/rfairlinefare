# Importar bibliotecas necess√°rias
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
#%%
# Importa os dados iniciais
entrada = pd.read_excel("Entrada_simples.xlsx", sheet_name="entrada")
entrada = pd.read_excel("Entrada_simples_v2.xlsx", sheet_name="entrada") # Base 1 (mercado 1 s/ tarifa e hor·rio)
entrada = pd.read_excel("Entrada_simples_v2.xlsx", sheet_name="entrada2") # Base 2 (mercado 1 + 2)
entrada = pd.read_excel("Entrada_simples_v2.xlsx", sheet_name="entrada3") # Base 3 (mercado 2)
#%%
# Remover todas as linhas com valores NA
dado_limpo = entrada.dropna()

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


#%%##############################################################################

# Separar features e vari√°vel alvo
# BKP    X = dado_limpo.drop(columns=["CM0", "ORDEM", "TIME", "TIME.1"])  # Features
X = dado_limpo.drop(columns=["CM0"])  # Features
y = dado_limpo["CM0"]  # Vari√°vel alvo

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123, stratify=y)

# Hiperpar‚metros Random Forest
modelo_rf = RandomForestClassifier(
    n_estimators=500, 
    random_state=123, 
    max_depth=25, 
    min_samples_split=4, 
    max_features='sqrt',
    n_jobs=-1
)


modelo_rf.fit(X_train, y_train)

# Gerar previs√µes
previsoes_rf = modelo_rf.predict(X_test)

#%%##############################################################################
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

#%%##################################################################################
# AVALIA√á√ÉO DO MODELO #
# MATRIZ DE CONFUS√ÉO #

# Definir as classes de 0 a 12
classes = list(range(13))

# Gerar a matriz de confus√£o
matriz_confusao = confusion_matrix(y_test, previsoes_rf, labels=classes)

# Verificar a matriz de confus√£o para garantir que esteja correta
print(matriz_confusao)

# Criar DataFrame a partir da matriz de confus√£o para plottar o gr√°fico
conf_df = pd.DataFrame(matriz_confusao, index=classes, columns=classes)

# Plotar o heatmap com cores e anota√ß√µes ajustadas
plt.figure(figsize=(10, 8))
sns.heatmap(conf_df, annot=True, fmt='d', cmap='Blues', cbar=False, 
            annot_kws={"color": "black"})  # Cor das anota√ß√µes

# Ajustar os t√≠tulos e r√≥tulos
plt.title('Matriz de Confus„o', fontsize=16)
plt.xlabel('Previsto', fontsize=12)
plt.ylabel('Observado', fontsize=12)
plt.xticks(rotation=45)
plt.yticks(rotation=0)

# Exibir o gr√°fico
plt.tight_layout()
plt.show()

####################################################################################
#%%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import numpy as np

# Avalia√ß√£o do modelo

# Acur√°cia
acuracia = accuracy_score(y_test, previsoes_rf)
print(f"Acur√°cia: {acuracia:.4f}")
#%%
# Contagem de inst‚ncias por classe
from collections import Counter
pesos = np.array(list(Counter(y_test).values()))

# Precis√£o, Sensibilidade (Recall) e F1-Score para cada classe
precisao = precision_score(y_test, previsoes_rf, average=None)
sensibilidade = recall_score(y_test, previsoes_rf, average=None)
f1_scores = f1_score(y_test, previsoes_rf, average=None)

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
print(classification_report(y_test, previsoes_rf, target_names=[str(c) for c in classes]))

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
#%%
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
previsoes_rf_CGR = modelo_rf.predict(X_CGR)

# Gerar a matriz de confus√£o
matriz_confusaoCGR = confusion_matrix(y_CGR, previsoes_rf_CGR, labels=classes)

# Verificar a matriz de confus√£o para garantir que esteja correta
print(matriz_confusaoCGR)

# Acur√°cia
acuraciaCGR = accuracy_score(y_CGR, previsoes_rf_CGR)
print(f"Acur√°cia: {acuraciaCGR:.4f}")

# Criar DataFrame a partir da matriz de confus√£o para plottar o gr√°fico
conf_dfCGR = pd.DataFrame(matriz_confusaoCGR, index=classes, columns=classes)

# Plotar o heatmap com cores e anota√ß√µes ajustadas
plt.figure(figsize=(10, 8))
sns.heatmap(conf_dfCGR, annot=True, fmt='d', cmap='Blues', cbar=False, 
            annot_kws={"color": "black"})  # Cor das anota√ß√µes


#%% ESTIMAR ACUR¡CIA PONDERADA (ERROS ADJACENTES REDUZIDOS)

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

#%%##########################################################################
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

# Suponha que 'y_test' e 'previsoes_rf' sejam arrays de n√∫meros inteiros (0, 1, 2, etc.)
# Reverter os inteiros para as classes originais
y_test_classes_CGR = reverter_para_classes(y_CGR)  # Classes reais
previsoes_rf_classes_CGR = reverter_para_classes(previsoes_rf_CGR)  # Previs√µes

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
previsoes_rf_mix_group_cgr = [map_to_mix_group(label) for label in previsoes_rf_classes_CGR]

# Calcular a acur√°cia total por grupo de mix
acuracia_mix_cgr = accuracy_score(y_test_mix_group_cgr, previsoes_rf_mix_group_cgr)
print(f"Acur√°cia por grupo de mix: {acuracia_mix:.4f}")

# Calcular a acur√°cia separada para cada grupo (MIX HIGH, MIX MID, MIX LOW)
for mix_group in ['MIX HIGH', 'MIX MID', 'MIX LOW']:
    y_test_group_cgr = [1 if label == mix_group else 0 for label in y_test_mix_group_cgr]
    previsoes_rf_group_cgr = [1 if label == mix_group else 0 for label in previsoes_rf_mix_group_cgr]
    
    acuracia_grupo = accuracy_score(y_test_group_cgr, previsoes_rf_group_cgr)
    print(f"Acur√°cia para {mix_group}: {acuracia_grupo:.4f}")
    
 #############
#%%
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

 # Suponha que 'y_test' e 'previsoes_rf' sejam arrays de n√∫meros inteiros (0, 1, 2, etc.)
 # Reverter os inteiros para as classes originais
 y_test_classes = reverter_para_classes(y_test)  # Classes reais
 previsoes_rf_classes = reverter_para_classes(previsoes_rf)  # Previs√µes

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
 previsoes_rf_mix_group = [map_to_mix_group(label) for label in previsoes_rf_classes]

 # Calcular a acur√°cia total por grupo de mix
 acuracia_mix = accuracy_score(y_test_mix_group, previsoes_rf_mix_group)
 print(f"Acur√°cia por grupo de mix: {acuracia_mix:.4f}")

 # Calcular a acur√°cia separada para cada grupo (MIX HIGH, MIX MID, MIX LOW)
 for mix_group in ['MIX HIGH', 'MIX MID', 'MIX LOW']:
     y_test_group = [1 if label == mix_group else 0 for label in y_test_mix_group]
     previsoes_rf_group = [1 if label == mix_group else 0 for label in previsoes_rf_mix_group]
     
     acuracia_grupo = accuracy_score(y_test_group, previsoes_rf_group)
     print(f"Acur√°cia para {mix_group}: {acuracia_grupo:.4f}")   
     
     
#%%###########################

X_test.to_excel("X_test_export.xlsx", index=False)  # O index=False remove o √≠ndice do arquivo
y_test.to_excel("y_test_export.xlsx", index=False)  # O index=False remove o √≠ndice do arquivo
previsoes_rf_series.to_excel("previsoes_rf_testexport.xlsx", index=False)  # O index=False remove o √≠ndice do arquivo 

previsoes_rf_series = pd.Series(previsoes_rf) 

#%%
# # RODADAS DE TESTE PARA HIPERPAR¬METROS

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# EspaÁo de busca para o Random Forest
rf_params = {
    'n_estimators': [100, 500, 1000],
    'max_depth': [10, 25, 50],
    'min_samples_split': [2, 4, 8],
    'max_features': ['sqrt', 'log2', None]
}

# Para o Random Forest
rf_model = RandomForestClassifier(random_state=123)
rf_grid_search = GridSearchCV(estimator=rf_model, param_grid=rf_params, cv=5, scoring='accuracy', n_jobs=-1)

# Executar para o Random Forest
rf_grid_search.fit(X_train, y_train)
print("Melhores par‚metros para o Random Forest:", rf_grid_search.best_params_)

## Melhores par‚metros para o Random Forest: {'max_depth': 25, 'max_features': None, 'min_samples_split': 4, 'n_estimators': 1000}
#%%
import numpy as np
from sklearn.metrics import top_k_accuracy_score

# Obter as probabilidades de previs„o para cada classe
y_pred_proba = modelo_rf.predict_proba(X_test)

# Definir o valor de K
K = 3

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
y_pred_proba = modelo_rf.predict_proba(X_test)

# Ordenar as probabilidades em cada linha em ordem crescente e selecionar as Top-K
sorted_probas = np.sort(y_pred_proba, axis=1)[:, -K:]

# Inverter para obter as Top-K probabilidades em ordem decrescente (do mais prov·vel ao menos prov·vel)
top_k_probas = sorted_probas[:, ::-1]

# Criar um DataFrame para mostrar as primeiras 10 linhas com as 3 probabilidades mais altas para cada linha
top_k_df = pd.DataFrame(top_k_probas[:10], columns=[f"Top-{i+1} Probability" for i in range(K)])

# Exibir o DataFrame
print("Top-3 probabilidades para as primeiras 10 linhas:")
print(top_k_df)
#%% ANTIGO (n„o usar)
import numpy as np

# Suponha que `y_test` e `previsoes_rf` sejam arrays contendo as classes verdadeiras e as classes previstas
# Certifique-se de que `y_test` e `previsoes_rf` s„o do tipo numpy array
y_test = np.array(y_test)  # Converte `y_test` para numpy array, se necess·rio
previsoes_rf = np.array(previsoes_rf)  # Converte `previsoes_rf` para numpy array, se necess·rio

# Filtrar Ìndices onde houve erro
erro_indices = np.where(y_test != previsoes_rf)[0]

# Calcular a diferenÁa absoluta entre a classe correta e a prevista para esses erros
erro_distancia_rf = np.abs(y_test[erro_indices] - previsoes_rf[erro_indices])

# Calcular a mÈdia e o desvio padr„o da dist‚ncia dos erros
mean_distancia_erro = np.mean(erro_distancia_rf)
std_distancia_erro = np.std(erro_distancia_rf)
qtd_erro = len(erro_indices)

# ConfiguraÁ„o das faixas (bins) para as dist‚ncias dos erros
bins = np.arange(1, max(erro_distancia_rf)+2, 1)

def calcular_frequencia_erro(erro_distancia_rf, bins):
    freq, _ = np.histogram(erro_distancia_rf, bins=bins)
    return freq

freq_rf = calcular_frequencia_erro(erro_distancia_rf, bins)

print(f"Quantidade de erros: {qtd_erro}")
print(f"MÈdia da dist‚ncia dos erros: {mean_distancia_erro:.4f}")
print(f"Desvio padr„o da dist‚ncia dos erros: {std_distancia_erro:.4f}")
print(freq_rf)
#%% DEFINITIVOOO
import numpy as np

# Suponha que `y_test` e `previsoes_rf` sejam arrays contendo as classes verdadeiras e as classes previstas
# Certifique-se de que `y_test` e `previsoes_rf` s„o do tipo numpy array
y_test = np.array(y_test)  # Converte `y_test` para numpy array, se necess·rio
previsoes_rf = np.array(previsoes_rf)  # Converte `previsoes_rf` para numpy array, se necess·rio

# Filtrar Ìndices onde houve erro
erro_indices = np.where(y_test != previsoes_rf)[0]

# Calcular a diferenÁa absoluta entre a classe correta e a prevista para esses erros
erro_distancia_rf = np.abs(y_test[erro_indices] - previsoes_rf[erro_indices])

# Calcular a mÈdia e o desvio padr„o da dist‚ncia dos erros
mean_distancia_erro = np.mean(erro_distancia_rf)
std_distancia_erro = np.std(erro_distancia_rf)
qtd_erro = len(erro_indices)

# ConfiguraÁ„o das faixas (bins) para as dist‚ncias dos erros, limitando a 12 bins
bins = np.arange(1, 13, 1)

def calcular_frequencia_erro(erro_distancia_rf, bins):
    freq, _ = np.histogram(erro_distancia_rf, bins=bins)
    # Ajustar para 12 posiÁıes, adicionando zeros onde necess·rio
    freq_ajustado = np.zeros(12, dtype=int)
    freq_ajustado[:len(freq)] = freq
    return freq_ajustado

freq_rf = calcular_frequencia_erro(erro_distancia_rf, bins)

print(f"Quantidade de erros: {qtd_erro}")
print(f"MÈdia da dist‚ncia dos erros: {mean_distancia_erro:.4f}")
print(f"Desvio padr„o da dist‚ncia dos erros: {std_distancia_erro:.4f}")
print(freq_rf)
#%%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Suponha que `y_test` e `previsoes_rf` sejam arrays contendo as classes verdadeiras e as classes previstas
# Certifique-se de que `y_test` e `previsoes_rf` s„o do tipo numpy array
y_test = np.array(y_test)  # Converte `y_test` para numpy array, se necess·rio
previsoes_rf = np.array(previsoes_rf)  # Converte `previsoes_rf` para numpy array, se necess·rio

# Filtrar Ìndices onde houve erro
erro_indices = np.where(y_test != previsoes_rf)[0]

# Calcular a diferenÁa absoluta entre a classe correta e a prevista para esses erros
erro_distancia_rf = np.abs(y_test[erro_indices] - previsoes_rf[erro_indices])

# ConfiguraÁ„o das faixas (bins) para as dist‚ncias dos erros, limitando a 12 bins
bins = np.arange(1, 13, 1)

def calcular_frequencia_erro(erro_distancia_rf, bins):
    freq, _ = np.histogram(erro_distancia_rf, bins=bins)
    # Ajustar para 12 posiÁıes, adicionando zeros onde necess·rio
    freq_ajustado = np.zeros(12, dtype=int)
    freq_ajustado[:len(freq)] = freq
    return freq_ajustado

freq_rf = calcular_frequencia_erro(erro_distancia_rf, bins)

# Dados para o gr·fico
distancias = np.arange(1, 13)
frequencias = freq_rf
#frequencias = np.array([456, 196, 140, 111, 59, 56, 32, 30, 15, 8, 2, 0])

# Gr·fico de barras com Seaborn
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.barplot(x=distancias, y=frequencias, color="blue", edgecolor="black")
#plt.title('Histograma de FrequÍncias por Dist‚ncia', fontsize=14)
plt.xlabel('Dist‚ncia entre Classe Correta e Prevista', fontsize=12)
plt.ylabel('FrequÍncia', fontsize=12)
plt.xticks(distancias - 1, fontsize=10)
plt.yticks(fontsize=10)
plt.ylim(0, 600)
plt.tight_layout()
plt.show()
#%%
import pickle
# Salvar `previsoes_rf`
with open('previsoes_rf.pkl', 'wb') as f:
    pickle.dump(previsoes_rf, f)
    
    #%% CURVA DE DENSIDADE SOZINHA
    
    # Plotar a curva de densidade baseada em freq_rf
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

# Dados simulados
distancias = np.arange(1, 13)
frequencias = freq_rf

# Expandir os dados para refletir as frequÍncias
dados_expandido = np.repeat(distancias, frequencias)

# Gr·fico de densidade
sns.kdeplot(data=dados_expandido, color="red", linewidth=2, label="Curva de Densidade")

# ConfiguraÁıes do gr·fico
plt.xlabel('Dist‚ncia entre Classe Correta e Prevista', fontsize=12)
plt.ylabel('Densidade', fontsize=12)
plt.title('Curva de Densidade baseada nas FrequÍncias', fontsize=14)
plt.xticks(distancias, fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.legend()
plt.show()


#%% COMBINA«√O FINAL

# ConfiguraÁ„o do estilo
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

# Criar o gr·fico e adicionar o histograma (barras)
fig, ax1 = plt.subplots(figsize=(10, 6))

# Gr·fico de barras
ax1.bar(distancias, frequencias, color="lightblue", edgecolor="black", linewidth = 1, label="FrequÍncia", width=1)
ax1.set_xlabel('Dist‚ncia entre Classe Correta e Prevista', fontsize=12)
ax1.set_ylabel('FrequÍncia', fontsize=12, color="black")
ax1.tick_params(axis='y', labelcolor="black")
ax1.set_ylim(0, 400)  # Limite superior para as colunas

# Adicionar o eixo secund·rio para a curva de densidade
ax2 = ax1.twinx()
sns.kdeplot(data=dados_expandido, color="red", linewidth=2, label="Curva de Densidade", ax=ax2, clip=(1, None))  # Ajuste para iniciar no ponto onde a densidade È maior que zero
ax2.set_ylabel('Densidade', fontsize=12, color="red")
ax2.tick_params(axis='y', labelcolor="red")
ax2.set_ylim(0, 0.5)  # Limite superior para o eixo da densidade

# Adicionar tÌtulo
plt.title('Histograma de FrequÍncias com Curva de Densidade - Random Forest (domÌnio 1 e 2)', fontsize=14)
fig.tight_layout()
# Mostrar rÛtulos de 2 em 2
plt.xticks(distancias[::1], fontsize=10)

# Remover as linhas horizontais
ax1.grid(False)
ax2.grid(False)

# Exibir a legenda
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

# Mostrar o gr·fico
plt.show()


#%%
import numpy as np
from sklearn.metrics import accuracy_score

# Dicion·rio de mapeamento reverso (inteiros para classes)
mapeamento_reverso = {
    0: 'Y', 1: 'B', 2: 'H', 3: 'K', 4: 'M', 5: 'L', 6: 'V', 
    7: 'X', 8: 'S', 9: 'N', 10: 'Q', 11: 'O', 12: 'G'
}

# FunÁ„o para aplicar o mapeamento reverso
def reverter_para_classes(numeros):
    return [mapeamento_reverso[numero] for numero in numeros]

# Suponha que 'y_test' e 'previsoes_rf' sejam arrays de n˙meros inteiros (0, 1, 2, etc.)
# Reverter os inteiros para as classes originais
y_test_classes = reverter_para_classes(y_test)  # Classes reais
previsoes_rf_classes = reverter_para_classes(previsoes_rf)  # Previsıes

# Definir os grupos de mix
mix_high = ['Y', 'B', 'H', 'K']  # 0, 1, 2, 3
mix_mid = ['M', 'L', 'V', 'X', 'S', 'N']  # 4, 5, 6, 7, 8, 9
mix_low = ['Q', 'O', 'G']  # 10, 11, 12

# FunÁ„o para mapear as classes para seus respectivos grupos
def map_to_mix_group(label):
    if label in mix_high:
        return 'MIX HIGH'
    elif label in mix_mid:
        return 'MIX MID'
    elif label in mix_low:
        return 'MIX LOW'
    else:
        return 'UNKNOWN'  # Caso tenha algum valor inesperado

# Aplicar o mapeamento para os grupos de mix nas classes reais e previstas
y_test_mix_group = [map_to_mix_group(label) for label in y_test_classes]
previsoes_rf_mix_group = [map_to_mix_group(label) for label in previsoes_rf_classes]

# Calcular a acur·cia total por grupo de mix
acuracia_mix = accuracy_score(y_test_mix_group, previsoes_rf_mix_group)
print(f"Acur·cia por grupo de mix: {acuracia_mix:.4f}")

# Calcular a acur·cia separada para cada grupo (MIX HIGH, MIX MID, MIX LOW)
for mix_group in ['MIX HIGH', 'MIX MID', 'MIX LOW']:
    # Filtrar as previsıes e valores reais para o grupo atual
    y_test_filtered = [1 if label == mix_group else 0 for label in y_test_mix_group]
    previsoes_rf_filtered = [1 if label == mix_group else 0 for label in previsoes_rf_mix_group]
    
    # Calcular a acur·cia para o grupo especÌfico
    acuracia_grupo = accuracy_score(y_test_filtered, previsoes_rf_filtered)
    print(f"Acur·cia para {mix_group}: {acuracia_grupo:.4f}")
#%%

# Suponha que y_test_filtered e previsoes_rf_filtered j· estejam definidos
data = {
    'y_test_filtered': y_test_mix_group,
    'previsoes_rf_filtered': previsoes_rf_mix_group
}

# Criar o DataFrame e exportar para Excel
df = pd.DataFrame(data)
df.to_excel("y_test_previsoes_rf_filtered.xlsx", index=False)
print("Arquivo Excel 'y_test_previsoes_rf_filtered.xlsx' foi criado com sucesso.")

#%%
np.save('previsoes_rf.npy', previsoes_rf)