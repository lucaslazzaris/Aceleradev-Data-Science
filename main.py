#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[3]:


black_friday.sample(5)


# In[4]:


black_friday.info()


# In[5]:


black_friday.describe()


# In[6]:


black_friday.isna().sum()


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[7]:


def q1():
    # answer (537577 , 12)
    (n_observations,n_columns) = black_friday.shape
    return (n_observations, n_columns)


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[8]:


def q2():
    # Answer 49348
    mask_female = black_friday['Gender'] == "F"
    mask_age = black_friday['Age'] == "26-35"
    N_young_woman = black_friday[mask_female & mask_age].shape[0]
    return N_young_woman


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[9]:


def q3():
    # Answer 5891
    return black_friday['User_ID'].nunique()


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[10]:


def q4():
    # Answer 3
    return black_friday.dtypes.nunique()


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[11]:


def q5():
    # Answer 0.3055897108693266
    observations_total = black_friday.shape[0]
    observations_wo_na = black_friday.dropna().shape[0]
    return 1 - observations_wo_na / observations_total 


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[12]:


def q6():
    # Answer 373299
    return  black_friday.isnull().sum().max()


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[13]:


def q7():
    # Answer 16.0.
    return black_friday["Product_Category_3"].value_counts().idxmax()


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[15]:


def q8():
    # Answer 0.3925748592124437
    min_at_0 = black_friday["Purchase"] -  black_friday["Purchase"].min()
    max_difference = (black_friday["Purchase"].max() - black_friday["Purchase"].min())
    normalized_purchase = min_at_0 / max_difference
    return normalized_purchase.mean()


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[ ]:


def q9():
    # Answer 0.3925748592124437
    purchase = black_friday["Purchase"]
    normalized_purchase = (purchase - purchase.mean()) / (purchase.std())
    mask = (normalized_purchase >= -1) & (normalized_purchase <= 1)
    return normalized_purchase[mask].shape[0]


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[31]:


def q10():
    # Answer True
    null_in_both = ((black_friday["Product_Category_2"].isnull()) & (black_friday["Product_Category_3"].isnull())).sum()
    null_in_2_total = black_friday["Product_Category_2"].isnull().sum()
    return bool(null_in_both == null_in_2_total)

