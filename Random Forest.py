# Importar bibliotecas necessárias
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
#%%
# Importa os dados iniciais
entrada = pd.read_excel("Entrada_simples.xlsx", sheet_name="entrada")
entrada = pd.read_excel("Entrada_simples_v2.xlsx", sheet_name="entrada") # Base 1 (mercado 1 s/ tarifa e horário)
entrada = pd.read_excel("Entrada_simples_v2.xlsx", sheet_name="entrada2") # Base 2 (mercado 1 + 2)
entrada = pd.read_excel("Entrada_simples_v2.xlsx", sheet_name="entrada3") # Base 3 (mercado 2)
#%%
# Remover todas as linhas com valores NA
dado_limpo = entrada.dropna()

#%%##############################################################################

# Separar features e variável alvo
X = dado_limpo.drop(columns=["CM0"])  # Features
y = dado_limpo["CM0"]  # VariÃ¡vel alvo

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123, stratify=y)

# Hiperparâmetros Random Forest
modelo_rf = RandomForestClassifier(
    n_estimators=500, 
    random_state=123, 
    max_depth=25, 
    min_samples_split=4, 
    max_features='sqrt',
    n_jobs=-1
)


modelo_rf.fit(X_train, y_train)

# Gerar previsões
previsoes_rf = modelo_rf.predict(X_test)

#%%##############################################################################
# K-Fold Test

from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np

# Definir o StratifiedKFold cross-validator
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

# Avaliar o modelo com K-fold Cross Validation usando estratificação
scores = cross_val_score(modelo_rf, X, y, cv=skf, scoring='accuracy')

# Exibir as pontuações e a média
print("Scores em cada fold: ", scores)
print("Acurácia média: ", np.mean(scores))

#%%##################################################################################
# AVALIAÃ‡ÃƒO DO MODELO #
# MATRIZ DE CONFUSÃƒO #

# Definir as classes de 0 a 12
classes = list(range(13))

# Gerar a matriz de confusÃ£o
matriz_confusao = confusion_matrix(y_test, previsoes_rf, labels=classes)

# Verificar a matriz de confusÃ£o para garantir que esteja correta
print(matriz_confusao)

# Criar DataFrame a partir da matriz de confusÃ£o para plottar o grÃ¡fico
conf_df = pd.DataFrame(matriz_confusao, index=classes, columns=classes)

# Plotar o heatmap com cores e anotaÃ§Ãµes ajustadas
plt.figure(figsize=(10, 8))
sns.heatmap(conf_df, annot=True, fmt='d', cmap='Blues', cbar=False, 
            annot_kws={"color": "black"})  # Cor das anotaÃ§Ãµes

# Ajustar os títulos e rótulos
plt.title('Matriz de Confusão', fontsize=16)
plt.xlabel('Previsto', fontsize=12)
plt.ylabel('Observado', fontsize=12)
plt.xticks(rotation=45)
plt.yticks(rotation=0)

# Exibir o gráfico
plt.tight_layout()
plt.show()

####################################################################################
#%%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import numpy as np

# Avaliação do modelo

# Acurácia
acuracia = accuracy_score(y_test, previsoes_rf)
print(f"AcurÃ¡cia: {acuracia:.4f}")
#%%
# Contagem de instâncias por classe
from collections import Counter
pesos = np.array(list(Counter(y_test).values()))

# Precisão, Sensibilidade (Recall) e F1-Score para cada classe
precisao = precision_score(y_test, previsoes_rf, average=None)
sensibilidade = recall_score(y_test, previsoes_rf, average=None)
f1_scores = f1_score(y_test, previsoes_rf, average=None)

# Exibir as métricas para cada classe
for i, classe in enumerate(classes):
    print(f"\nClasse: {classe}")
    print(f"PrecisÃ£o: {precisao[i]:.4f}")
    print(f"Sensibilidade (Recall): {sensibilidade[i]:.4f}")
    print(f"F1-Score: {f1_scores[i]:.4f}")
    
    # Cálculo da média ponderada para Precisão, Sensibilidade e F1-Score
    precisao_ponderada = np.average(precisao, weights=pesos)
    sensibilidade_ponderada = np.average(sensibilidade, weights=pesos)
    f1_score_ponderado = np.average(f1_scores, weights=pesos)    
        
    print("\nMédias Ponderadas:")
    print(f"Precisão Ponderada: {precisao_ponderada:.4f}")
    print(f"Sensibilidade Ponderada: {sensibilidade_ponderada:.4f}")
    print(f"F1-Score Ponderado: {f1_score_ponderado:.4f}") 

# Relatório de classificação completo
print("\nRelatÃ³rio de ClassificaÃ§Ã£o Completo:\n")
print(classification_report(y_test, previsoes_rf, target_names=[str(c) for c in classes]))

# Para visualizar a precisão, recall e F1-score de maneira tabular
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

X_CGR = dado_limpoCGR.drop(columns=["CM0"])  # Features
y_CGR = dado_limpoCGR["CM0"]  # VariÃ¡vel alvo

# Alinhar colunas entre o modelo treinado (X_train) e a nova base (X_CGR)
X_CGR = X_CGR.reindex(columns=X_train.columns, fill_value=0)


# Gerar previsÃµes CGR
previsoes_rf_CGR = modelo_rf.predict(X_CGR)

# Gerar a matriz de confusÃ£o
matriz_confusaoCGR = confusion_matrix(y_CGR, previsoes_rf_CGR, labels=classes)

# Verificar a matriz de confusÃ£o para garantir que esteja correta
print(matriz_confusaoCGR)

# AcurÃ¡cia
acuraciaCGR = accuracy_score(y_CGR, previsoes_rf_CGR)
print(f"AcurÃ¡cia: {acuraciaCGR:.4f}")

# Criar DataFrame a partir da matriz de confusão para plottar o gráfico
conf_dfCGR = pd.DataFrame(matriz_confusaoCGR, index=classes, columns=classes)

# Plotar o heatmap com cores e anotações ajustadas
plt.figure(figsize=(10, 8))
sns.heatmap(conf_dfCGR, annot=True, fmt='d', cmap='Blues', cbar=False, 
            annot_kws={"color": "black"})  # Cor das anotaÃ§Ãµes


#%% ESTIMAR ACURÁCIA PONDERADA (ERROS ADJACENTES REDUZIDOS)

import numpy as np  # Importar NumPy

# Inicializar uma variÃ¡vel para a acurÃ¡cia ponderada
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

print(f"Acurácia ponderada: {weighted_accuracy:.4f}")

#%%##########################################################################
###########################################################################
# ACURÁCIA POR AGRUPAMENTO CGR

import numpy as np
from sklearn.metrics import accuracy_score

# Função para mapear as classes para seus respectivos grupos
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

# Calcular a acurÃ¡cia total por grupo de mix
acuracia_mix_cgr = accuracy_score(y_test_mix_group_cgr, previsoes_rf_mix_group_cgr)
print(f"AcurÃ¡cia por grupo de mix: {acuracia_mix:.4f}")

# Calcular a acurÃ¡cia separada para cada grupo (MIX HIGH, MIX MID, MIX LOW)
for mix_group in ['MIX HIGH', 'MIX MID', 'MIX LOW']:
    y_test_group_cgr = [1 if label == mix_group else 0 for label in y_test_mix_group_cgr]
    previsoes_rf_group_cgr = [1 if label == mix_group else 0 for label in previsoes_rf_mix_group_cgr]
    
    acuracia_grupo = accuracy_score(y_test_group_cgr, previsoes_rf_group_cgr)
    print(f"AcurÃ¡cia para {mix_group}: {acuracia_grupo:.4f}")
    
 #############
#%%
# TESTE Comum   agrupamento 
    
 import numpy as np
 from sklearn.metrics import accuracy_score

 # Função para mapear as classes para seus respectivos grupos
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

 # Calcular a acurácia total por grupo de mix
 acuracia_mix = accuracy_score(y_test_mix_group, previsoes_rf_mix_group)
 print(f"AcurÃ¡cia por grupo de mix: {acuracia_mix:.4f}")

 # Calcular a acurácia separada para cada grupo (MIX HIGH, MIX MID, MIX LOW)
 for mix_group in ['MIX HIGH', 'MIX MID', 'MIX LOW']:
     y_test_group = [1 if label == mix_group else 0 for label in y_test_mix_group]
     previsoes_rf_group = [1 if label == mix_group else 0 for label in previsoes_rf_mix_group]
     
     acuracia_grupo = accuracy_score(y_test_group, previsoes_rf_group)
     print(f"AcurÃ¡cia para {mix_group}: {acuracia_grupo:.4f}")   
     
     
#%%###########################

X_test.to_excel("X_test_export.xlsx", index=False)  # O index=False remove o Ã­ndice do arquivo
y_test.to_excel("y_test_export.xlsx", index=False)  # O index=False remove o Ã­ndice do arquivo
previsoes_rf_series.to_excel("previsoes_rf_testexport.xlsx", index=False)  # O index=False remove o Ã­ndice do arquivo 

previsoes_rf_series = pd.Series(previsoes_rf) 

#%%
# # RODADAS DE TESTE PARA HIPERPARÂMETROS

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Espaço de busca para o Random Forest
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
print("Melhores parâmetros para o Random Forest:", rf_grid_search.best_params_)

## Melhores parâmetros para o Random Forest: {'max_depth': 25, 'max_features': None, 'min_samples_split': 4, 'n_estimators': 1000}
#%%
import numpy as np
from sklearn.metrics import top_k_accuracy_score

# Obter as probabilidades de previsão para cada classe
y_pred_proba = modelo_rf.predict_proba(X_test)

# Definir o valor de K
K = 3

# Calcular o Top-K Accuracy
top_k_acc = top_k_accuracy_score(y_test, y_pred_proba, k=K)
print(f"Top-{K} Accuracy: {top_k_acc:.4f}")


# Ordenar as probabilidades para cada instância de forma decrescente e selecionar as Top-K
sorted_probas = np.sort(y_pred_proba, axis=1)[:, -K:]
# Inverter a ordem para que as probabilidades estejam do mais provável ao menos provável
top_k_probas = sorted_probas[:, ::-1]

# Calcular a mediana das probabilidades para a 1ª, 2ª e 3ª escolhas
median_proba_1st = np.median(top_k_probas[:, 0])
median_proba_2nd = np.median(top_k_probas[:, 1])
median_proba_3rd = np.median(top_k_probas[:, 2])

print(f"Mediana da probabilidade da 1ª escolha: {median_proba_1st:.4f}")
print(f"Mediana da probabilidade da 2ª escolha: {median_proba_2nd:.4f}")
print(f"Mediana da probabilidade da 3ª escolha: {median_proba_3rd:.4f}")
#%%
import numpy as np
import pandas as pd

# Definir o valor de K (Top-K)
K = 3

# Obter as probabilidades previstas para cada classe
y_pred_proba = modelo_rf.predict_proba(X_test)

# Ordenar as probabilidades em cada linha em ordem crescente e selecionar as Top-K
sorted_probas = np.sort(y_pred_proba, axis=1)[:, -K:]

# Inverter para obter as Top-K probabilidades em ordem decrescente (do mais provável ao menos provável)
top_k_probas = sorted_probas[:, ::-1]

# Criar um DataFrame para mostrar as primeiras 10 linhas com as 3 probabilidades mais altas para cada linha
top_k_df = pd.DataFrame(top_k_probas[:10], columns=[f"Top-{i+1} Probability" for i in range(K)])

# Exibir o DataFrame
print("Top-3 probabilidades para as primeiras 10 linhas:")
print(top_k_df)
#%% ANTIGO (não usar)
import numpy as np

# Suponha que `y_test` e `previsoes_rf` sejam arrays contendo as classes verdadeiras e as classes previstas
# Certifique-se de que `y_test` e `previsoes_rf` são do tipo numpy array
y_test = np.array(y_test)  # Converte `y_test` para numpy array, se necessário
previsoes_rf = np.array(previsoes_rf)  # Converte `previsoes_rf` para numpy array, se necessário

# Filtrar índices onde houve erro
erro_indices = np.where(y_test != previsoes_rf)[0]

# Calcular a diferença absoluta entre a classe correta e a prevista para esses erros
erro_distancia_rf = np.abs(y_test[erro_indices] - previsoes_rf[erro_indices])

# Calcular a média e o desvio padrão da distância dos erros
mean_distancia_erro = np.mean(erro_distancia_rf)
std_distancia_erro = np.std(erro_distancia_rf)
qtd_erro = len(erro_indices)

# Configuração das faixas (bins) para as distâncias dos erros
bins = np.arange(1, max(erro_distancia_rf)+2, 1)

def calcular_frequencia_erro(erro_distancia_rf, bins):
    freq, _ = np.histogram(erro_distancia_rf, bins=bins)
    return freq

freq_rf = calcular_frequencia_erro(erro_distancia_rf, bins)

print(f"Quantidade de erros: {qtd_erro}")
print(f"Média da distância dos erros: {mean_distancia_erro:.4f}")
print(f"Desvio padrão da distância dos erros: {std_distancia_erro:.4f}")
print(freq_rf)
#%% DEFINITIVOOO
import numpy as np

# Suponha que `y_test` e `previsoes_rf` sejam arrays contendo as classes verdadeiras e as classes previstas
# Certifique-se de que `y_test` e `previsoes_rf` são do tipo numpy array
y_test = np.array(y_test)  # Converte `y_test` para numpy array, se necessário
previsoes_rf = np.array(previsoes_rf)  # Converte `previsoes_rf` para numpy array, se necessário

# Filtrar índices onde houve erro
erro_indices = np.where(y_test != previsoes_rf)[0]

# Calcular a diferença absoluta entre a classe correta e a prevista para esses erros
erro_distancia_rf = np.abs(y_test[erro_indices] - previsoes_rf[erro_indices])

# Calcular a média e o desvio padrão da distância dos erros
mean_distancia_erro = np.mean(erro_distancia_rf)
std_distancia_erro = np.std(erro_distancia_rf)
qtd_erro = len(erro_indices)

# Configuração das faixas (bins) para as distâncias dos erros, limitando a 12 bins
bins = np.arange(1, 13, 1)

def calcular_frequencia_erro(erro_distancia_rf, bins):
    freq, _ = np.histogram(erro_distancia_rf, bins=bins)
    # Ajustar para 12 posições, adicionando zeros onde necessário
    freq_ajustado = np.zeros(12, dtype=int)
    freq_ajustado[:len(freq)] = freq
    return freq_ajustado

freq_rf = calcular_frequencia_erro(erro_distancia_rf, bins)

print(f"Quantidade de erros: {qtd_erro}")
print(f"Média da distância dos erros: {mean_distancia_erro:.4f}")
print(f"Desvio padrão da distância dos erros: {std_distancia_erro:.4f}")
print(freq_rf)
#%%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Suponha que `y_test` e `previsoes_rf` sejam arrays contendo as classes verdadeiras e as classes previstas
# Certifique-se de que `y_test` e `previsoes_rf` são do tipo numpy array
y_test = np.array(y_test)  # Converte `y_test` para numpy array, se necessário
previsoes_rf = np.array(previsoes_rf)  # Converte `previsoes_rf` para numpy array, se necessário

# Filtrar índices onde houve erro
erro_indices = np.where(y_test != previsoes_rf)[0]

# Calcular a diferença absoluta entre a classe correta e a prevista para esses erros
erro_distancia_rf = np.abs(y_test[erro_indices] - previsoes_rf[erro_indices])

# Configuração das faixas (bins) para as distâncias dos erros, limitando a 12 bins
bins = np.arange(1, 13, 1)

def calcular_frequencia_erro(erro_distancia_rf, bins):
    freq, _ = np.histogram(erro_distancia_rf, bins=bins)
    # Ajustar para 12 posições, adicionando zeros onde necessário
    freq_ajustado = np.zeros(12, dtype=int)
    freq_ajustado[:len(freq)] = freq
    return freq_ajustado

freq_rf = calcular_frequencia_erro(erro_distancia_rf, bins)

# Dados para o gráfico
distancias = np.arange(1, 13)
frequencias = freq_rf
#frequencias = np.array([456, 196, 140, 111, 59, 56, 32, 30, 15, 8, 2, 0])

# Gráfico de barras com Seaborn
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.barplot(x=distancias, y=frequencias, color="blue", edgecolor="black")
#plt.title('Histograma de Frequências por Distância', fontsize=14)
plt.xlabel('Distância entre Classe Correta e Prevista', fontsize=12)
plt.ylabel('Frequência', fontsize=12)
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

# Expandir os dados para refletir as frequências
dados_expandido = np.repeat(distancias, frequencias)

# Gráfico de densidade
sns.kdeplot(data=dados_expandido, color="red", linewidth=2, label="Curva de Densidade")

# Configurações do gráfico
plt.xlabel('Distância entre Classe Correta e Prevista', fontsize=12)
plt.ylabel('Densidade', fontsize=12)
plt.title('Curva de Densidade baseada nas Frequências', fontsize=14)
plt.xticks(distancias, fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.legend()
plt.show()


#%% COMBINAÇÃO FINAL

# Configuração do estilo
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

# Criar o gráfico e adicionar o histograma (barras)
fig, ax1 = plt.subplots(figsize=(10, 6))

# Gráfico de barras
ax1.bar(distancias, frequencias, color="lightblue", edgecolor="black", linewidth = 1, label="Frequência", width=1)
ax1.set_xlabel('Distância entre Classe Correta e Prevista', fontsize=12)
ax1.set_ylabel('Frequência', fontsize=12, color="black")
ax1.tick_params(axis='y', labelcolor="black")
ax1.set_ylim(0, 400)  # Limite superior para as colunas

# Adicionar o eixo secundário para a curva de densidade
ax2 = ax1.twinx()
sns.kdeplot(data=dados_expandido, color="red", linewidth=2, label="Curva de Densidade", ax=ax2, clip=(1, None))  # Ajuste para iniciar no ponto onde a densidade é maior que zero
ax2.set_ylabel('Densidade', fontsize=12, color="red")
ax2.tick_params(axis='y', labelcolor="red")
ax2.set_ylim(0, 0.5)  # Limite superior para o eixo da densidade

# Adicionar título
plt.title('Histograma de Frequências com Curva de Densidade - Random Forest (domínio 1 e 2)', fontsize=14)
fig.tight_layout()
# Mostrar rótulos de 2 em 2
plt.xticks(distancias[::1], fontsize=10)

# Remover as linhas horizontais
ax1.grid(False)
ax2.grid(False)

# Exibir a legenda
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

# Mostrar o gráfico
plt.show()


#%%
import numpy as np
from sklearn.metrics import accuracy_score

# Função para mapear as classes para seus respectivos grupos
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

# Calcular a acurácia total por grupo de mix
acuracia_mix = accuracy_score(y_test_mix_group, previsoes_rf_mix_group)
print(f"Acurácia por grupo de mix: {acuracia_mix:.4f}")

# Calcular a acurácia separada para cada grupo (MIX HIGH, MIX MID, MIX LOW)
for mix_group in ['MIX HIGH', 'MIX MID', 'MIX LOW']:
    # Filtrar as previsões e valores reais para o grupo atual
    y_test_filtered = [1 if label == mix_group else 0 for label in y_test_mix_group]
    previsoes_rf_filtered = [1 if label == mix_group else 0 for label in previsoes_rf_mix_group]
    
    # Calcular a acurácia para o grupo específico
    acuracia_grupo = accuracy_score(y_test_filtered, previsoes_rf_filtered)
    print(f"Acurácia para {mix_group}: {acuracia_grupo:.4f}")
#%%

# Suponha que y_test_filtered e previsoes_rf_filtered já estejam definidos
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
