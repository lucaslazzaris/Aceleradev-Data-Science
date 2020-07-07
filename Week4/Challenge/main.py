#!/usr/bin/env python
# coding: utf-8

# # Desafio 3
# 
# Neste desafio, iremos praticar nossos conhecimentos sobre distribuições de probabilidade. Para isso,
# dividiremos este desafio em duas partes:
#     
# 1. A primeira parte contará com 3 questões sobre um *data set* artificial com dados de uma amostra normal e
#     uma binomial.
# 2. A segunda parte será sobre a análise da distribuição de uma variável do _data set_ [Pulsar Star](https://archive.ics.uci.edu/ml/datasets/HTRU2), contendo 2 questões.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF


# In[3]:


#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# ## Parte 1

# ### _Setup_ da parte 1

# In[4]:


np.random.seed(42)
    
dataframe = pd.DataFrame({"normal": sct.norm.rvs(20, 4, size=10000),
                     "binomial": sct.binom.rvs(100, 0.2, size=10000)})


# ## Inicie sua análise a partir da parte 1 a partir daqui

# In[5]:


# Sua análise da parte 1 começa aqui.
dataframe.head()


# In[6]:


sns.distplot(dataframe['normal'])
sns.distplot(dataframe['binomial'], bins=range(0, 35))


# In[7]:


dataframe.describe()


# In[8]:


dataframe.quantile(0.25)['normal']


# In[ ]:





# ## Questão 1
# 
# Qual a diferença entre os quartis (Q1, Q2 e Q3) das variáveis `normal` e `binomial` de `dataframe`? Responda como uma tupla de três elementos arredondados para três casas decimais.
# 
# Em outra palavras, sejam `q1_norm`, `q2_norm` e `q3_norm` os quantis da variável `normal` e `q1_binom`, `q2_binom` e `q3_binom` os quantis da variável `binom`, qual a diferença `(q1_norm - q1 binom, q2_norm - q2_binom, q3_norm - q3_binom)`?

# In[12]:


def q1():
    q_norm = dataframe['normal'].quantile(q=[0.25, 0.5, 0.75])
    q_binom = dataframe['binomial'].quantile(q=[0.25, 0.5, 0.75])
    
    return tuple(round(qi_norm - qi_binom, 3) for qi_norm, qi_binom in zip(q_norm, q_binom))


# Para refletir:
# 
# * Você esperava valores dessa magnitude?
# 
# * Você é capaz de explicar como distribuições aparentemente tão diferentes (discreta e contínua, por exemplo) conseguem dar esses valores?

# Minhas resposta:
# 
# * Sim, a soma de diversas operações com p% de ocorrencia tende a uma normal, pelo teorema do limite central

# ## Questão 2
# 
# Considere o intervalo $[\bar{x} - s, \bar{x} + s]$, onde $\bar{x}$ é a média amostral e $s$ é o desvio padrão. Qual a probabilidade nesse intervalo, calculada pela função de distribuição acumulada empírica (CDF empírica) da variável `normal`? Responda como uma único escalar arredondado para três casas decimais.

# In[56]:


def q2():
    # Creating ECDF Object (It will return the CDF from emprical data)
    ecdf = ECDF(dataframe['normal'])

    lower_bound = dataframe['normal'].mean() - dataframe['normal'].std()
    upper_bound = dataframe['normal'].mean() + dataframe['normal'].std()
    
    # Probability for the lower and upper_bound
    p1 = ecdf(lower_bound)
    p2 = ecdf(upper_bound)
    return float(round(p2 - p1, 3))


# Para refletir:
# 
# * Esse valor se aproxima do esperado teórico?
# * Experimente também para os intervalos $[\bar{x} - 2s, \bar{x} + 2s]$ e $[\bar{x} - 3s, \bar{x} + 3s]$.

# Minha resposta
# 
# * Tanto para 1s, 2s e 3s há grande concordância com o valor esperado, demostrando a robustez do ECDF implementado pela biblioteca statsmodel

# ## Questão 3
# 
# Qual é a diferença entre as médias e as variâncias das variáveis `binomial` e `normal`? Responda como uma tupla de dois elementos arredondados para três casas decimais.
# 
# Em outras palavras, sejam `m_binom` e `v_binom` a média e a variância da variável `binomial`, e `m_norm` e `v_norm` a média e a variância da variável `normal`. Quais as diferenças `(m_binom - m_norm, v_binom - v_norm)`?

# In[52]:


def q3():
    # Retorne aqui o resultado da questão 3.
    m_binom = dataframe['binomial'].mean()
    m_norm = dataframe['normal'].mean()
    v_binom = dataframe['binomial'].var()
    v_norm = dataframe['normal'].var()
    return round(m_binom - m_norm, 3), round(v_binom - v_norm, 3)


# Para refletir:
# 
# * Você esperava valore dessa magnitude?
# * Qual o efeito de aumentar ou diminuir $n$ (atualmente 100) na distribuição da variável `binomial`?

# Minha resposta:
# * Valores de pequena magnitude eram de fato esperados
# * Ao alterar **n**, estamos alterando os valores máximos da distribuição, quando aumentamos **n** aumentamos a média e o desvio padrão da distribuição binomial, e quando reduzimos **n**, reduzimos a média e desvio padrão
# * Portanto, alterar **n** se não de uma maneira muito sensível, tende a aumentar a diferença entre os valores da distribuição normal ($\mu=20,\sigma=4$) e da binomal

# ## Parte 2

# ### _Setup_ da parte 2

# In[14]:


stars = pd.read_csv("pulsar_stars.csv")

stars.rename({old_name: new_name
              for (old_name, new_name)
              in zip(stars.columns,
                     ["mean_profile", "sd_profile", "kurt_profile", "skew_profile", "mean_curve", "sd_curve", "kurt_curve", "skew_curve", "target"])
             },
             axis=1, inplace=True)

stars.loc[:, "target"] = stars.target.astype(bool)


# ## Inicie sua análise da parte 2 a partir daqui

# In[15]:


# Sua análise da parte 2 começa aqui.
stars.head()


# In[16]:


stars.describe()


# In[17]:


false_pulsar = stars.query('target == 0')['mean_profile']
false_pulsar_mean_profile_standardized = (false_pulsar - false_pulsar.mean()) / false_pulsar.std()
sns.distplot(false_pulsar_mean_profile_standardized)


# ## Questão 4
# 
# Considerando a variável `mean_profile` de `stars`:
# 
# 1. Filtre apenas os valores de `mean_profile` onde `target == 0` (ou seja, onde a estrela não é um pulsar).
# 2. Padronize a variável `mean_profile` filtrada anteriormente para ter média 0 e variância 1.
# 
# Chamaremos a variável resultante de `false_pulsar_mean_profile_standardized`.
# 
# Encontre os quantis teóricos para uma distribuição normal de média 0 e variância 1 para 0.80, 0.90 e 0.95 através da função `norm.ppf()` disponível em `scipy.stats`.
# 
# Quais as probabilidade associadas a esses quantis utilizando a CDF empírica da variável `false_pulsar_mean_profile_standardized`? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[20]:


def q4():
    false_pulsar = stars.query('target == 0')['mean_profile']
    false_pulsar_mean_profile_standardized = sct.zscore(false_pulsar)
    
    theorical_quartiles = sct.norm.ppf([0.8, 0.9, 0.95])
    ecdf = ECDF(false_pulsar_mean_profile_standardized)
    experimental_quartiles = tuple(float(round(exp_quartil, 3)) for exp_quartil in ecdf(theorical_quartiles))
    return experimental_quartiles

q4()


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?

# Minhas respostas:
# * Os valores encontrados fazem sentido, e aproximam-se de uma normal, pelo fato de que aparentemente a distribuição de estralas não pulsantes ser normal para variável de mean_profile

# ## Questão 5
# 
# Qual a diferença entre os quantis Q1, Q2 e Q3 de `false_pulsar_mean_profile_standardized` e os mesmos quantis teóricos de uma distribuição normal de média 0 e variância 1? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[83]:


def q5():
    false_pulsar = stars.query('target == 0')['mean_profile']
    false_pulsar_mean_profile_standardized = (false_pulsar - false_pulsar.mean()) / false_pulsar.std()
    
    quartiles =  [0.25, 0.50, 0.75]
    q_stars = [false_pulsar_mean_profile_standardized.quantile(i) for i in quartiles]
    q_norm = [sct.norm.ppf(i) for i in quartiles]

    return tuple(float(round(q_stars[i] - q_norm[i], 3)) for i in range(len(quartiles)))


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?
# * Curiosidade: alguns testes de hipóteses sobre normalidade dos dados utilizam essa mesma abordagem.

# # Minhas respostas:
# 
# * A Distribuição da variável é próxima à normal padrão, espcialmente a partir do segundo quartil, no primeiro, ainda há um devio da ordem de 10% (de 0.25 para 0.282 na inversa da função cumulativa de probabilidade)

# In[ ]:




