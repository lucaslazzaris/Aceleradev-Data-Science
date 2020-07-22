#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[167]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer


# In[115]:


# Algumas configurações para o matplotlib.
#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[116]:


countries = pd.read_csv("countries.csv")


# In[117]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head(5)


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# In[118]:


countries.shape


# In[119]:


countries.dtypes


# In[120]:


# Sua análise começa aqui.
num_vars = ["Pop_density", "Coastline_ratio", "Net_migration", "Infant_mortality",
            "Literacy", "Phones_per_1000", "Arable", "Crops", "Other", "Climate",
            "Birthrate", "Deathrate", "Agriculture", "Industry", "Service"]

for var in num_vars:
    try:
        countries[var] = pd.to_numeric(countries[var].str.replace(',', '.'))
    except:
        print(var)


# In[122]:


num_vars = [col for col, dtype in zip(countries.columns, countries.dtypes) if dtype == np.int64 or dtype == np.float64]


# In[123]:


strip_vars = ["Country", "Region"]
for var in strip_vars:
    try:
        countries[var] = countries[var].str.strip()
    except:
        print(var)


# In[124]:


countries.dtypes


# In[125]:


countries.head()


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[151]:


def q1():
    unique_regions = countries['Region'].unique()
    return list(np.sort(unique_regions))


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[127]:


def q2():
    discretizer = KBinsDiscretizer(n_bins = 10, encode='ordinal', strategy='quantile')
    discretizer.fit(countries[['Pop_density']])
    pop_density = discretizer.transform(countries[['Pop_density']]).astype(np.int)
    return int(pop_density[pop_density == 9].size)


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[226]:


def q3():
    Nregions = countries['Region'].nunique() + countries['Region'].isnull().any().sum()
    Nclimate = countries['Climate'].nunique() + countries['Climate'].isnull().any().sum()
    return int(Nregions + Nclimate)

q3()


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[129]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[149]:


def q4():
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy='median')),
        ('std_scaler', StandardScaler())
    ])
    
    num_pipeline.fit(countries[num_vars])
    
    test_data = np.array(test_country)[2:] # Ignore first two string columns
    test_data = test_data.reshape(1, -1) # Just because sklearn says so...
    return float(round(num_pipeline.transform(test_data)[0,9], 3)) # Arable is the 9th element   


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[163]:


def q5():
    quantiles = countries['Net_migration'].quantile([0.25, 0.75])
    quantile_len = quantiles.iloc[1] - quantiles.iloc[0]
    
    max_value_allowed = quantiles.iloc[1] + 1.5 * quantile_len
    min_value_allowed = quantiles.iloc[0] - 1.5 * quantile_len
    
    mask_above = (countries['Net_migration'] > max_value_allowed)
    mask_below = (countries['Net_migration'] < min_value_allowed)
    
    outliers_above = countries['Net_migration'][mask_above]
    outliers_below = countries['Net_migration'][mask_below]
    return int(outliers_below.size), int(outliers_above.size), False


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[166]:


categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)


# In[219]:


def q6():
    
    count_vectorizer = CountVectorizer()
    newsgroup_counts = count_vectorizer.fit_transform(newsgroup.data)
    index = int(count_vectorizer.vocabulary_['phone'])
    
    phone_df = pd.DataFrame(newsgroup_counts[:, index].toarray())
    return int(phone_df.sum())


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[220]:


def q7():
    tfidf_vectorizer = TfidfVectorizer()
    newsgroups_tfidf_vectorized  = tfidf_vectorizer.fit_transform(newsgroup.data)
    index = int(tfidf_vectorizer.vocabulary_['phone'])

    phone_df = pd.DataFrame(newsgroups_tfidf_vectorized[:, index].toarray())
    return round(float(phone_df.sum()), 3)


# In[ ]:




