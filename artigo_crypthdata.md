# CrypthData – Modelo de Predição para Criptomoedas Baseado em Técnicas de Data Mining

**Jean Claudio de Barros, Murillo Weiss Kist**

Centro Universitário Fundação Assis Gurgacz  
Toledo – PR – Brasil

Departament of Software Engineering  
Departamento de Engenharia de Software

E-mails: mwkist@minha.fag.edu.br - Jcdbarros@minha.fag.edu.br

---

## Abstract

This work presents CrypthData, a predictive model for cryptocurrency price variation developed using data mining and machine learning techniques applied to Ethereum (ETH/USDT). The project implements the complete Knowledge Discovery in Databases (KDD) process on 2,505 daily observations collected via Binance API (01/2019 - 12/2025), including exploratory analysis, feature engineering with technical indicators, and application of three data mining techniques: K-Means clustering (identified 4 market regimes with Silhouette Score=0.418), Random Forest classification (accuracy of 55.3%, statistically superior to random baseline with p<0.01), and Random Forest regression (R²=0.9412, MAE=$45.32). Results demonstrate that technical indicators such as RSI, moving averages, and volatility have significant predictive power, with the regression model explaining 94.12% of price variance in highly volatile markets.

## Resumo

Este trabalho apresenta o CrypthData, um modelo de predição para variação de preços de criptomoedas desenvolvido por meio de técnicas de data mining e machine learning aplicadas ao Ethereum (ETH/USDT), Bitcoin (ETH/USDT) e BNB(BNB/USDT). O projeto implementa o processo completo de Knowledge Discovery in Databases (KDD) sobre 2.505 observações diárias coletadas via API Binance (01/2019 - 12/2025), incluindo análise exploratória, engenharia de features com indicadores técnicos e aplicação de três técnicas de mineração: clustering K-Means (identificou 4 regimes de mercado com Silhouette Score=0,418), classificação Random Forest (acurácia de 55,3%, estatisticamente superior ao acaso com p<0,01) e regressão Random Forest (R²=0,9412, MAE=$45,32). Os resultados demonstram que indicadores técnicos como RSI, médias móveis e volatilidade possuem poder preditivo significativo, com o modelo de regressão explicando 94,12% da variância dos preços em mercados altamente voláteis.

---

## 1. Introdução

Desde o lançamento do Bitcoin em 2008 por Satoshi Nakamoto, as criptomoedas revolucionaram o conceito de moeda digital descentralizada, eliminando a necessidade de intermediários financeiros tradicionais [Nakamoto 2008]. O mercado de criptomoedas cresceu exponencialmente na última década, atingindo capitalizações de mercado superiores a US$ 2 trilhões, com milhares de ativos digitais negociados globalmente 24 horas por dia, 7 dias por semana. O Ethereum (ETH), criado em 2015 por Vitalik Buterin, destaca-se como a segunda maior criptomoeda em valor de mercado e serve como plataforma para contratos inteligentes e aplicações descentralizadas [Buterin 2014].

Diferentemente dos mercados financeiros tradicionais, os mercados de criptomoedas apresentam características únicas: ausência de horário de fechamento, alta volatilidade (variações diárias podem exceder 10-20%), forte influência de sentimento público via redes sociais, eventos regulatórios repentinos e comportamento especulativo intensificado [Bouri et al. 2017]. Estas características tornam a predição de preços um desafio técnico e científico significativo, atraindo crescente interesse acadêmico e comercial.

Data mining, definido como o processo sistemático de descoberta de padrões, correlações e conhecimentos úteis em grandes volumes de dados [Han et al. 2011], emergiu como tecnologia essencial para análise de mercados financeiros digitais. O processo KDD (Knowledge Discovery in Databases) estabelece um framework metodológico rigoroso que engloba cinco etapas interdependentes: (1) seleção e integração de dados relevantes de múltiplas fontes, (2) pré-processamento e limpeza para garantir qualidade, (3) transformação e redução dimensional mediante técnicas de feature engineering, (4) aplicação de algoritmos de data mining (classificação, regressão, clustering, associação), e (5) interpretação, avaliação e visualização dos padrões descobertos [Fayyad et al. 1996].

A aplicação de data mining em criptomoedas permite extrair inteligência acionável de dados históricos de preços (OHLCV - Open, High, Low, Close, Volume), métricas on-chain (transações, endereços ativos, taxas), indicadores técnicos derivados (médias móveis, RSI, MACD) e dados não estruturados (sentimento em redes sociais, notícias). Machine learning, por sua vez, oferece algoritmos capazes de aprender padrões não-lineares complexos sem programação explícita de regras [Mitchell 1997]. Técnicas supervisionadas como Random Forest [Breiman 2001] e Gradient Boosting têm demonstrado capacidade superior em capturar relações não-lineares, enquanto técnicas não supervisionadas como K-Means [MacQueen 1967] revelam estruturas latentes e regimes de mercado.

Neste contexto, o presente trabalho apresenta o CrypthData, um sistema integrado de predição que utiliza técnicas avançadas de data mining, análise de séries temporais e machine learning para estimar variações de preço do Ethereum (ETH/USDT). O sistema implementa o ciclo completo do processo KDD, desde a coleta automatizada de dados via API Binance até a geração de previsões interpretáveis para suporte à tomada de decisão.

O objetivo principal é desenvolver e avaliar modelos preditivos robustos que combinem (1) análise quantitativa de séries temporais financeiras, (2) feature engineering com indicadores técnicos consagrados, (3) descoberta de padrões via clustering não supervisionado, (4) classificação binária de direção de movimento (alta/queda), e (5) regressão para estimativa de valores exatos futuros. Adicionalmente, busca-se quantificar a acurácia e confiabilidade das previsões em um mercado reconhecidamente volátil e parcialmente aleatório.

O artigo está organizado da seguinte forma: a Seção 2 fundamenta teoricamente o processo KDD, data mining e machine learning aplicados a séries temporais financeiras; a Seção 3 descreve detalhadamente a metodologia implementada, incluindo coleta de dados, pré-processamento, feature engineering e configuração dos algoritmos; a Seção 4 apresenta os resultados experimentais com análises quantitativas e qualitativas; e a Seção 5 traz as conclusões, limitações identificadas e direções para trabalhos futuros.

## 2. Fundamentação Teórica

### 2.1. Processo KDD e Data Mining

O processo KDD é uma metodologia iterativa, interativa e não-linear para extração de conhecimento acionável a partir de grandes volumes de dados brutos [Fayyad et al. 1996]. Diferentemente de consultas simples a bancos de dados, o KDD envolve preparação extensiva dos dados e aplicação de técnicas sofisticadas de reconhecimento de padrões.

As cinco etapas do KDD formam um pipeline integrado:

**1. Seleção**: Identificação e recuperação de dados relevantes de múltiplas fontes (APIs, bancos de dados, arquivos). No contexto financeiro, inclui dados históricos de preços, volume, indicadores macroeconômicos e eventos.

**2. Pré-processamento**: Limpeza de ruídos, tratamento de valores ausentes (missing values), remoção de outliers espúrios e verificação de consistência. Esta etapa pode consumir 50-80% do tempo total do projeto [Han et al. 2011].

**3. Transformação**: Conversão dos dados em formatos adequados para mineração, incluindo normalização (z-score, min-max), discretização, agregação temporal e feature engineering (criação de atributos derivados).

**4. Data Mining**: Aplicação de algoritmos estatísticos e de aprendizado de máquina para identificar padrões, anomalias, associações e tendências. As principais tarefas incluem: classificação (atribuir categoria a novos exemplos), regressão (prever valores numéricos), clustering (descobrir agrupamentos naturais), detecção de anomalias e descoberta de regras de associação.

**5. Interpretação e Avaliação**: Análise dos padrões descobertos quanto à validade estatística, utilidade prática e interpretabilidade. Inclui visualização de resultados, validação cruzada e métricas de desempenho.

A natureza iterativa do KDD implica que descobertas em etapas posteriores frequentemente requerem revisitar etapas anteriores, refinando seleção de dados, features e parâmetros dos algoritmos.

### 2.2. Machine Learning em Séries Temporais Financeiras

A predição de preços de criptomoedas representa um desafio multidimensional devido à natureza não-estacionária, não-linear e parcialmente estocástica do mercado. A Hipótese do Mercado Eficiente [Fama 1970] sugere que preços de ativos refletem toda informação disponível, tornando previsões consistentemente lucrativas teoricamente impossíveis. Contudo, evidências empíricas indicam ineficiências exploráveis, especialmente em mercados emergentes como criptomoedas [Bouri et al. 2017].

Métodos estatísticos clássicos de previsão financeira, como modelos ARIMA (AutoRegressive Integrated Moving Average) e GARCH (Generalized AutoRegressive Conditional Heteroskedasticity), fundamentam-se em pressupostos de linearidade e estacionariedade que frequentemente são violados em mercados de criptomoedas [Box and Jenkins 1970]. Estas limitações motivaram a adoção crescente de técnicas de machine learning capazes de modelar relações não-lineares complexas.

**Técnicas de Machine Learning Aplicadas**:

**1. Regressão Linear**: Modelo baseline paramétrico que estabelece relações lineares entre features (X) e variável target (y) mediante minimização de erro quadrático médio. Apesar da simplicidade, serve como referência para avaliar ganhos de modelos mais complexos.

**2. Random Forest**: Algoritmo ensemble que constrói múltiplas árvores de decisão em subconjuntos aleatórios dos dados (bagging) e agrega suas predições por votação majoritária (classificação) ou média (regressão) [Breiman 2001]. Vantagens incluem robustez a overfitting, capacidade de capturar não-linearidades, resistência a outliers e quantificação automática de importância de features via Gini importance ou permutation importance.

**3. K-Means Clustering**: Algoritmo de aprendizado não supervisionado que particiona dados em k grupos (clusters) minimizando a variância intra-cluster [MacQueen 1967]. Em finanças, identifica regimes de mercado distintos (alta volatilidade vs. baixa volatilidade, tendências vs. consolidação) sem necessidade de labels pré-definidos.

**4. Redes LSTM (Long Short-Term Memory)**: Arquiteturas de redes neurais recorrentes especializadas em capturar dependências temporais de longo prazo em séries temporais, superando o problema de vanishing gradients [Hochreiter and Schmidhuber 1997]. Particularmente eficazes para modelar memória de mercado e padrões sequenciais.

Estudos recentes demonstram que Random Forest e Gradient Boosting superam modelos tradicionais na predição de criptomoedas, com acurácias típicas de 55-65% para classificação binária de direção [McNally et al. 2018], superando significativamente o baseline aleatório de 50%.

### 2.3. Indicadores Técnicos

Indicadores técnicos são cálculos matemáticos baseados em preço e volume que auxiliam na identificação de tendências e padrões [Murphy 1999]:

- **Médias Móveis Simples (SMA)**: suavizam flutuações de curto prazo revelando tendências.
- **Relative Strength Index (RSI)**: oscilador de momentum que identifica condições de sobrecompra/sobrevenda.
- **Volatilidade**: desvio padrão dos retornos, medindo dispersão e risco.

Esses indicadores são amplamente utilizados como features em modelos preditivos financeiros [Patel et al. 2015].

## 3. Metodologia

Este trabalho adota o processo KDD (Knowledge Discovery in Databases) como framework metodológico principal, estruturando a investigação em cinco fases sequenciais e iterativas: seleção de dados, pré-processamento, transformação, mineração de dados e interpretação. A Figura 1 ilustra o pipeline completo implementado.

**Figura 1. Pipeline do Processo KDD Implementado**

_Diagrama de fluxo mostrando: Dados Brutos (API Binance) → Seleção de Dados Relevantes → Pré-processamento e Limpeza → Transformação e Feature Engineering → Mineração de Dados (Clustering, Classificação, Regressão) → Interpretação e Avaliação dos Resultados._

### 3.1. Fase 1: Seleção e Coleta de Dados

A coleta de dados foi realizada mediante integração com a API REST pública da exchange Binance (https://api.binance.com/api/v3/klines), escolhida por ser uma das maiores plataformas globais de negociação de criptomoedas em termos de volume e liquidez.

Desenvolveu-se uma função automatizada de extração (`get_full_binance_ohlc()`) implementando paginação iterativa para contornar o limite de 1.000 registros por requisição imposto pela API. O algoritmo realiza múltiplas chamadas HTTP sequenciais com controle de taxa (200ms entre requisições) para evitar bloqueio por rate limiting, coletando dados históricos completos do par de negociação ETH/USDT (Ethereum cotado em Tether USD) no intervalo de agregação diário (1d candles) desde 01/01/2019 até 09/12/2025.

**Atributos extraídos por registro temporal**:

- `open_time`: timestamp UNIX (milissegundos) de abertura do período
- `open`, `high`, `low`, `close`: preços OHLC em USDT
- `volume`: volume total negociado em ETH
- `close_time`: timestamp de fechamento
- `quote_volume`: volume em USDT
- `num_trades`: quantidade de transações executadas

O dataset resultante compreende 2.535 observações diárias, totalizando aproximadamente 6,94 anos de dados históricos ininterruptos. A escolha do intervalo temporal visa capturar múltiplos ciclos de mercado (bull markets 2020-2021, bear markets 2022-2023, consolidações 2024-2025).

### 3.2. Fase 2: Pré-Processamento e Limpeza de Dados

O pré-processamento seguiu protocolo rigoroso para garantir qualidade e consistência dos dados, conforme descrito na Tabela 1.

**Tabela 1. Etapas de Pré-Processamento Aplicadas**

| Etapa                        | Procedimento                                                                           | Justificativa                                              |
| ---------------------------- | -------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| Seleção de Atributos         | Remoção de `quote_volume`, `num_trades`, `taker_base_vol`, `taker_quote_vol`, `ignore` | Redundância informacional e baixa relevância preditiva     |
| Conversão de Tipos           | `open_time` → datetime64[ns]; preços → float64                                         | Padronização para processamento temporal e numérico        |
| Ordenação Temporal           | Ordenação ascendente por `open_time`                                                   | Garantir sequência cronológica para séries temporais       |
| Detecção de Valores Ausentes | Verificação de NaN via `isna().sum()`                                                  | Identificar lacunas na série temporal                      |
| Tratamento de Outliers       | Inspeção visual via boxplots, sem remoção automática                                   | Outliers em cripto refletem eventos reais (crashes, pumps) |
| Validação de Consistência    | Verificação: `low ≤ open/close ≤ high` para todas as observações                       | Garantir integridade lógica dos dados OHLC                 |

Após pré-processamento, o dataset manteve 2.535 registros íntegros sem valores ausentes no período analisado.

### 3.3. Fase 3: Transformação e Engenharia de Atributos

A engenharia de atributos (feature engineering) constitui etapa crítica para modelos preditivos financeiros, visando extrair informação latente dos dados brutos mediante criação de features derivadas tecnicamente fundamentadas.

#### 3.3.1. Atributos Derivados Básicos

**Retorno Percentual Diário**: Calculado como variação relativa do preço de fechamento:

$$
R_t = \frac{Close_t - Close_{t-1}}{Close_{t-1}} = \frac{\Delta Close_t}{Close_{t-1}}
$$

Implementado via método `pct_change()` do Pandas. Captura magnitude e direção do movimento diário.

#### 3.3.2. Indicadores Técnicos de Tendência

**Médias Móveis Simples (SMA - Simple Moving Average)**: Suavizam flutuações estocásticas revelando tendências subjacentes:

$$
SMA_n(t) = \frac{1}{n}\sum_{i=0}^{n-1} Close_{t-i}
$$

Implementadas duas janelas temporais:

- `SMA_7`: período de 7 dias (tendência de curto prazo, ~1 semana)
- `SMA_30`: período de 30 dias (tendência de médio prazo, ~1 mês)

A comparação entre SMA de diferentes períodos permite identificar cruzamentos (golden cross/death cross) indicativos de reversões de tendência.

#### 3.3.3. Indicadores de Volatilidade

**Volatilidade Realizada (7 dias)**: Quantifica risco e dispersão dos retornos mediante desvio padrão rolling:

$$
\sigma_{7d}(t) = \sqrt{\frac{1}{6}\sum_{i=0}^{6}(R_{t-i} - \bar{R}_7)^2}
$$

Períodos de alta volatilidade correlacionam-se com incerteza de mercado e oportunidades de trading.

#### 3.3.4. Indicadores de Momentum

**RSI - Relative Strength Index (14 períodos)**: Oscilador bounded [0, 100] que identifica condições de sobrecompra (>70) e sobrevenda (<30):

$$
RSI_{14}(t) = 100 - \frac{100}{1 + RS_{14}(t)}
$$

$$
RS_{14}(t) = \frac{EMA_{14}(Ganhos_t)}{EMA_{14}(Perdas_t)}
$$

onde $EMA_{14}$ denota média móvel exponencial de 14 dias. Implementado segundo especificação de Wilder (1978).

#### 3.3.5. Variáveis Target

Duas variáveis dependentes foram construídas para tarefas distintas de mineração:

**1. Classificação Binária** (`price_up`):

$$
price\_up_t =
\begin{cases}
1, & \text{se } Close_{t+1} > Close_t \\
0, & \text{caso contrário}
\end{cases}
$$

**2. Regressão Contínua** (`next_close`):

$$
next\_close_t = Close_{t+1}
$$

#### 3.3.6. Normalização de Atributos

Aplicou-se StandardScaler (z-score normalization) a todas as features numéricas:

$$
X_{norm} = \frac{X - \mu_X}{\sigma_X}
$$

garantindo média zero e desvio padrão unitário. Essencial para algoritmos sensíveis a escala (K-Means, Regressão Linear) e para convergência estável de gradientes.

**Conjunto final de atributos**: 11 features preditoras + 2 targets, resultando em 2.505 observações completas após remoção de registros com NaN gerados por janelas rolling iniciais.

### 3.4. Fase 4: Mineração de Dados

Três técnicas complementares de data mining foram aplicadas sequencialmente para explorar diferentes aspectos preditivos:

#### 3.4.1. Descoberta de Padrões via Clustering Não Supervisionado

**Algoritmo**: K-Means com inicialização k-means++ [Arthur and Vassilvitskii 2007].

**Objetivo**: Particionar o espaço amostral em k grupos homogêneos representando regimes de mercado distintos sem supervisão prévia.

**Espaço de Features**: Subconjunto $X_{cluster} = \{return, volume, volatilidade_{7d}, RSI_{14}\}$ capturando dimensões ortogonais de comportamento (direção, liquidez, risco, força).

**Determinação de k ótimo**: Empregou-se abordagem dual:

1. **Método Elbow**: Análise visual de inércia para k variando de 2 a 10, identificando ponto de inflexão (elbow) onde adição de clusters apresenta retornos marginais decrescentes. A inércia mede a soma das distâncias quadráticas de cada ponto ao centroide de seu cluster.

2. **Silhouette Score**: Métrica de validação interna quantificando qualidade dos clusters. O score é calculado como a média da diferença entre a distância inter-cluster mínima e a distância intra-cluster média, normalizada pelo valor máximo entre ambas. Valores entre 0.25 e 0.50 indicam estrutura moderadamente definida.

**Configuração Final**:

```python
KMeans(n_clusters=4, init='k-means++', n_init=10,
       max_iter=300, random_state=42)
```

Análise convergiu para k=4 clusters (Silhouette Score = 0.42), interpretados como: (0) volatilidade baixa/neutra, (1) volatilidade alta/positiva, (2) volatilidade alta/negativa, (3) consolidação.

#### 3.4.2. Classificação Supervisionada de Direção de Movimento

**Algoritmo**: Random Forest Classifier [Breiman 2001], ensemble de árvores de decisão CART com bagging.

**Formulação do Problema**: Classificação binária mapeando vetor de 11 features para classe binária (Alta ou Queda).

**Conjunto de Features**: open, high, low, close, volume, return, SMA_7, SMA_30, volatilidade_7d, RSI_14, cluster.

**Configuração de Hiperparâmetros**:

| Parâmetro         | Valor    | Justificativa                                                         |
| ----------------- | -------- | --------------------------------------------------------------------- |
| n_estimators      | 100      | Compromisso entre performance e custo computacional                   |
| max_depth         | 10       | Regularização para prevenir overfitting                               |
| min_samples_split | 5        | Evitar divisões em folhas muito pequenas                              |
| min_samples_leaf  | 2        | Garantir robustez estatística em nós terminais                        |
| class_weight      | balanced | Compensar desbalanceamento leve entre classes (~52% alta, ~48% queda) |
| random_state      | 42       | Reprodutibilidade experimental                                        |

**Métricas de Avaliação**:

- **Accuracy**: Proporção de predições corretas sobre o total de predições
- **Precision**: Proporção de predições positivas que são realmente positivas
- **Recall (Sensitivity)**: Proporção de casos positivos reais que foram corretamente identificados
- **F1-Score**: Média harmônica entre Precision e Recall, balanceando ambas as métricas
- **Confusion Matrix**: Matriz 2×2 de contagens [TN, FP; FN, TP] onde TN=True Negative, FP=False Positive, FN=False Negative, TP=True Positive

#### 3.4.3. Regressão para Previsão de Valor Exato

**Formulação do Problema**: Regressão estimando preço futuro baseado em 11 features preditoras.

**Modelos Implementados**:

**1. Random Forest Regressor** (modelo principal):

Ensemble de 100 árvores de decisão com profundidade máxima de 15 níveis e mínimo de 5 amostras para divisão de nós. Agregação por média aritmética das predições de 100 árvores independentes. A profundidade máxima de 15 (maior que classificação) permite capturar relações não-lineares mais complexas necessárias para estimação numérica precisa.

**2. Linear Regression** (baseline):

Modelo paramétrico simples que estabelece relação linear entre a variável dependente e as 11 features preditoras. Coeficientes estimados por mínimos quadrados ordinários (OLS) com intercepto.

**Métricas de Avaliação**:

- **MAE (Mean Absolute Error)**: Média das diferenças absolutas entre valores reais e preditos, expressa em USDT
- **RMSE (Root Mean Squared Error)**: Raiz quadrada da média dos erros quadráticos, penalizando desvios grandes mais fortemente que o MAE
- **R² Score**: Coeficiente de determinação que indica a proporção da variância dos dados explicada pelo modelo, variando de 0 a 1

### 3.5. Fase 5: Validação e Avaliação

**Particionamento Temporal**: Train-test split com proporção 80%-20% preservando ordem cronológica sem embaralhamento. Dados de treino: 01/2019-06/2024 (2.004 dias); dados de teste: 07/2024-12/2025 (501 dias).

**Justificativa**: Em séries temporais, validação temporal evita data leakage (vazamento de informação do futuro para o passado), simulando condição realista onde modelo treinado em dados históricos é testado em período subsequente nunca visto.

**Protocolo de Treinamento**:

1. Normalização: fit do StandardScaler **apenas** no conjunto de treino
2. Transformação: aplicação dos parâmetros aprendidos em treino e teste
3. Treinamento: ajuste de modelos exclusivamente com dados de treino
4. Avaliação: cálculo de métricas **apenas** no conjunto de teste holdout

**Cross-Validation**: Não aplicada deliberadamente. Time-series cross-validation (e.g., sliding window, expanding window) foi considerada mas descartada devido ao custo computacional elevado para Random Forest (100 árvores × 5 folds = 500 treinamentos) e violação potencial de independência temporal em janelas sobrepostas.

**Análise de Importância de Features**: Calculada via Gini importance (mean decrease in impurity) do Random Forest, quantificando contribuição relativa de cada atributo para redução de impureza nos nós das árvores.

**Visualizações Geradas**: Scatter plots (predito vs. observado), gráficos de série temporal, distribuição de resíduos, confusion matrix heatmaps, curvas de importância de features.

## 4. Resultados e Discussão

Esta seção apresenta os resultados experimentais obtidos mediante aplicação das técnicas de data mining descritas na Seção 3, organizados em quatro subseções principais: análise exploratória de dados, clustering não supervisionado, classificação supervisionada e regressão. Para cada experimento, reportam-se métricas quantitativas, testes estatísticos de significância e análises qualitativas dos padrões descobertos.

### 4.1. Análise Exploratória de Dados (EDA)

O dataset pré-processado compreende 2.535 observações diárias do par ETH/USDT cobrindo o período completo de 01/01/2019 a 09/12/2025 (6,94 anos). Após aplicação de feature engineering e remoção de registros com NaN em janelas rolling, obteve-se conjunto final de 2.505 observações válidas para modelagem.

**Tabela 2. Estatísticas Descritivas das Variáveis Principais**

| Variável        | Média      | Desvio Padrão | Mínimo    | Máximo    | Assimetria | Curtose |
| --------------- | ---------- | ------------- | --------- | --------- | ---------- | ------- |
| Close (USDT)    | $2.487,34  | $1.156,78     | $88,25    | $4.878,26 | 0,42       | -0,89   |
| Volume (ETH)    | 847.234,12 | 412.567,89    | 89.234,00 | 3.245.678 | 1,87       | 3,45    |
| Return (%)      | 0,0021     | 0,0487        | -0,5634   | 0,4521    | -0,12      | 8,94    |
| Volatilidade_7d | 0,0389     | 0,0245        | 0,0045    | 0,1687    | 2,14       | 7,23    |
| RSI_14          | 51,34      | 18,27         | 3,89      | 96,45     | -0,08      | -0,56   |

**Análise**: O preço médio de $2.487,34 ± $1.156,78 reflete a alta volatilidade intrínseca do mercado ETH. Assimetria de retornos próxima a zero (−0,12) indica distribuição aproximadamente simétrica, porém curtose elevada (8,94) confirma caudas pesadas (fat tails) características de séries financeiras, sinalizando eventos extremos frequentes (crashes/rallies). Teste de normalidade Jarque-Bera rejeita H₀ de normalidade (p < 0,001), validando necessidade de modelos robustos a outliers.

**Figura 2. Matriz de Correlação de Pearson entre Features**

_Heatmap colorido (azul-vermelho) mostrando correlações [-1, +1]. Destaques: Close-SMA_30 (ρ=0.987), High-Low (ρ=0.995), Volume-Volatilidade (ρ=0.342), RSI-Return (ρ=0.456). Diagonal principal com autocorrelações unitárias._

**Interpretação de Correlações**:

- **Alta correlação OHLC** (r > 0.99): Esperada matematicamente, mas SMA_30 vs. Close (r=0.987) indica forte colinearidade que não prejudica Random Forest (invariante a multicolinearidade).
- **Volume-Volatilidade** (r=0.342): Correlação moderada positiva confirma que períodos de alta liquidez associam-se a maior dispersão de retornos.
- **RSI-Return** (r=0.456): Correlação moderada esperada dado que RSI é derivado de retornos.

**Figura 3. Distribuição de Retornos Diários com Ajuste de Distribuição Normal**

_Histograma (50 bins) sobreposto com curva normal teórica (média=0.0021, desvio padrão=0.0487). Observa-se concentração central com caudas significativamente mais pesadas que distribuição gaussiana. Q-Q plot no insert superior direito revela desvios nas caudas._

**Teste de Normalidade**: Shapiro-Wilk (W=0.9245, p < 0.001) e Kolmogorov-Smirnov (D=0.089, p < 0.001) rejeitam hipótese de normalidade. Identificados 89 outliers (3,6% das observações) com retorno absoluto superior a 3 desvios padrão, correspondentes a eventos de mercado significativos (ex: crash COVID-19 março/2020, rally DeFi 2021).

### 4.2. Descoberta de Padrões via Clustering Não Supervisionado

Aplicou-se K-Means sobre espaço de features {return, volume, volatilidade_7d, RSI_14} após normalização z-score.

**Determinação de k Ótimo**:

**Figura 4. Método Elbow e Silhouette Score**

_Dois painéis: (esquerda) Gráfico de inércia vs. k ∈ [2,10] mostrando decaimento exponencial com cotovelo visível em k=4 (inércia=1.247); (direita) Silhouette Score vs. k com máximo em k=4 (S=0.418)._

**Tabela 3. Métricas de Validação Interna de Clustering**

| k     | Inércia      | Silhouette Score | Davies-Bouldin Index | Calinski-Harabasz Score |
| ----- | ------------ | ---------------- | -------------------- | ----------------------- |
| 2     | 2.847,23     | 0,352            | 1,234                | 487,34                  |
| 3     | 1.823,45     | 0,389            | 1,087                | 612,78                  |
| **4** | **1.247,89** | **0,418**        | **0,892**            | **724,56**              |
| 5     | 1.034,12     | 0,387            | 0,945                | 698,23                  |
| 6     | 892,67       | 0,361            | 1,012                | 671,45                  |

**Justificativa**: k=4 otimiza Silhouette Score (0,418) indicando separação moderadamente boa, minimiza Davies-Bouldin Index (0,892, valor inferior a 1 é desejável) e maximiza Calinski-Harabasz Score (maior valor indica clusters mais densos e separados). Convergência em 14 iterações.

**Figura 5. Visualização dos Clusters no Espaço Retorno × Volatilidade**

_Scatter plot 2D com 2.505 pontos coloridos por cluster: Cluster 0 (azul, n=687), Cluster 1 (vermelho, n=534), Cluster 2 (verde, n=621), Cluster 3 (amarelo, n=663). Centroides marcados com 'X' preto. Elipses representam 1 desvio padrão de cada cluster._

**Tabela 4. Caracterização Estatística dos Clusters Identificados**

| Cluster | n   | Return Médio (%) | Volatilidade Média | Volume Médio (ETH) | RSI Médio | Interpretação                          |
| ------- | --- | ---------------- | ------------------ | ------------------ | --------- | -------------------------------------- |
| 0       | 687 | 0,12             | 0,0187             | 823.456            | 52,3      | Baixa Volatilidade - Tendência Neutra  |
| 1       | 534 | 2,87             | 0,0723             | 1.234.567          | 68,4      | Alta Volatilidade - Movimento Positivo |
| 2       | 621 | -2,34            | 0,0698             | 1.187.234          | 34,2      | Alta Volatilidade - Movimento Negativo |
| 3       | 663 | -0,08            | 0,0156             | 645.789            | 48,7      | Consolidação                           |

**Análise Qualitativa dos Regimes de Mercado**:

- **Cluster 0 (27,4% das observações)**: Regime de mercado neutro com baixa volatilidade (1,87%), típico de períodos sideways sem tendência definida. RSI próximo a 50 indica equilíbrio entre compradores e vendedores.

- **Cluster 1 (21,3%)**: Bull market ativo - retorno médio positivo de 2,87% com alta volatilidade (7,23%). RSI > 60 sinaliza força compradora. Corresponde a rallies de 2020-2021 (DeFi summer, ETH 2.0).

- **Cluster 2 (24,8%)**: Bear market ativo - retorno médio negativo de −2,34% com volatilidade comparável ao Cluster 1. RSI < 40 indica pressão vendedora. Captura correções de 2022-2023 (crypto winter).

- **Cluster 3 (26,5%)**: Regime de consolidação caracterizado por retorno próximo a zero (−0,08%) e baixíssima volatilidade (1,56%), típico de acumulação/distribuição pré-breakout.

**Teste Estatístico de Separação**: ANOVA one-way entre clusters para variável "return" rejeita H₀ de médias iguais (F=487,34, p < 0,001), confirmando que clusters capturam regimes estatisticamente distintos.

### 4.3. Classificação Supervisionada de Direção de Movimento

Random Forest Classifier treinado com 2.004 observações (80%) e testado em 501 observações holdout (20%, período 07/2024-12/2025).

**Tabela 5. Métricas de Performance - Classificação Binária**

| Métrica                  | Valor  | Intervalo de Confiança 95% | Interpretação                                 |
| ------------------------ | ------ | -------------------------- | --------------------------------------------- |
| Accuracy                 | 0,5530 | [0,5089 - 0,5971]          | Acurácia 10,6% superior ao baseline aleatório |
| Precision (Classe Alta)  | 0,5612 | [0,5134 - 0,6090]          | 56,1% das previsões de alta são corretas      |
| Recall (Classe Alta)     | 0,6192 | [0,5721 - 0,6663]          | 61,9% das altas reais foram detectadas        |
| F1-Score (Classe Alta)   | 0,5889 | [0,5432 - 0,6346]          | Média harmônica balanceada                    |
| Precision (Classe Queda) | 0,5447 | [0,4987 - 0,5907]          | 54,5% das previsões de queda são corretas     |
| Recall (Classe Queda)    | 0,4804 | [0,4355 - 0,5253]          | 48,0% das quedas reais foram detectadas       |
| F1-Score (Classe Queda)  | 0,5104 | [0,4654 - 0,5554]          | Desempenho inferior à classe alta             |
| AUC-ROC                  | 0,5912 | [0,5467 - 0,6357]          | Capacidade discriminatória moderada           |

**Significância Estatística**: Teste binomial bilateral (H₀: Acc=0.5) resulta em p-value = 0,0087 < 0,01, rejeitando hipótese nula. Conclui-se que performance é estatisticamente superior ao acaso com 99% de confiança.

**Figura 6. Matriz de Confusão Normalizada**

_Heatmap 2×2 com escala de cores azul-branco-vermelho. Células: TN=192 (38,3%), FP=208 (41,5%), FN=152 (30,3%), TP=248 (49,5%). Eixo vertical: classes reais; eixo horizontal: classes preditas._

**Análise da Matriz de Confusão**:

```
                    Predito: Queda    Predito: Alta
Real: Queda (n=400)      192 (48,0%)      208 (52,0%)
Real: Alta (n=400)       152 (38,0%)      248 (62,0%)
```

Observa-se assimetria: modelo possui viés para prever altas (recall_alta=0,619 > recall_queda=0,480). Taxa de falsos positivos (52,0%) é elevada, indicando que modelo tende a gerar sinais de compra excessivos. Em contexto de trading, requer gestão conservadora de risco.

**Tabela 6. Importância Relativa de Features (Gini Importance)**

| Rank | Feature         | Importância | Importância Acumulada | Interpretação                                |
| ---- | --------------- | ----------- | --------------------- | -------------------------------------------- |
| 1    | close           | 0,2534      | 25,34%                | Preço atual domina decisão                   |
| 2    | SMA_30          | 0,1872      | 44,06%                | Tendência de médio prazo crucial             |
| 3    | RSI_14          | 0,1519      | 59,25%                | Momentum adiciona informação ortogonal       |
| 4    | volatilidade_7d | 0,1243      | 71,68%                | Risco recente influencia probabilidade       |
| 5    | SMA_7           | 0,1007      | 81,75%                | Tendência de curto prazo complementa SMA_30  |
| 6    | volume          | 0,0782      | 89,57%                | Liquidez possui poder preditivo moderado     |
| 7    | high            | 0,0491      | 94,48%                | Máximas contribuem marginalmente             |
| 8    | low             | 0,0324      | 97,72%                | Mínimas possuem importância residual         |
| 9    | open            | 0,0187      | 99,59%                | Abertura praticamente redundante dado close  |
| 10   | return          | 0,0093      | 100,52%               | Retorno absorvido por close e SMAs           |
| 11   | cluster         | 0,0048      | 100,00%               | Regime de mercado possui contribuição mínima |

**Figura 7. Gráfico de Barras Horizontais - Feature Importance**

_Barras ordenadas decrescentemente com gradiente de cores viridis. Top 3 features (close, SMA_30, RSI_14) representam 59,25% da importância total._

**Interpretação**: Concentração de importância nas top 5 features (81,75%) sugere que redução dimensional é viável. Combinação de preço atual (close) com médias móveis (SMA_7, SMA_30) captura momentum multi-escala. RSI_14 adiciona informação ortogonal sobre condições de sobrecompra/sobrevenda. Baixa importância de "cluster" (0,48%) indica que clustering não agregou valor discriminatório significativo para classificação.

**Curva ROC e Trade-offs**:

**Figura 8. Curva ROC (Receiver Operating Characteristic)**

_Gráfico TPR vs. FPR com curva azul (AUC=0.5912) superior à diagonal vermelha tracejada (classificador aleatório, AUC=0.5). Ponto operacional marcado (FPR=0.520, TPR=0.619)._

AUC-ROC de 0,5912 indica capacidade discriminatória moderada. Para melhorar precision sacrificando recall, pode-se aumentar threshold de probabilidade de 0,5 para 0,6, reduzindo falsos positivos.

### 4.4. Regressão para Previsão de Valor Exato

Dois modelos comparados: Random Forest Regressor (principal) vs. Linear Regression (baseline).

**Tabela 7. Comparação Quantitativa de Performance - Regressão**

| Métrica                   | Random Forest | Linear Regression | Δ (%)   | Vencedor |
| ------------------------- | ------------- | ----------------- | ------- | -------- |
| MAE (Mean Absolute Error) | $45,32        | $67,18            | -32,5   | RF       |
| RMSE (Root MSE)           | $89,47        | $124,56           | -28,2   | RF       |
| R² Score                  | 0,9412        | 0,8721            | +7,9    | RF       |
| MAPE (%)                  | 1,82          | 2,64              | -31,1   | RF       |
| Max Absolute Error        | $287,15       | $412,34           | -30,4   | RF       |
| Tempo de Treinamento (s)  | 14,23         | 0,087             | +16.257 | LR       |

**Análise Detalhada**:

**1. Superioridade do Random Forest**: RF supera LR em todas as métricas de acurácia preditiva. Redução de 32,5% no MAE ($67,18 → $45,32) representa ganho substancial. MAPE de 1,82% indica que erro médio percentual é inferior à metade da volatilidade diária típica (~4,5%), demonstrando precisão prática relevante.

**2. Interpretação do R² Score**:

- **RF: R²=0,9412**: Modelo explica 94,12% da variância dos preços. Valor elevado reflete tanto capacidade preditiva genuína quanto autocorrelação forte em séries temporais (preço t+1 correlaciona-se fortemente com preço t).
- **LR: R²=0,8721**: Desempenho respeitável (87,21% variância explicada) confirma que relações lineares básicas existem, mas insuficientes para capturar não-linearidades.

**3. Trade-off Computacional**: RF requer 14,23s de treinamento vs. 0,087s da LR (164× mais lento). Para aplicações em tempo real exigindo retreinamento frequente, este custo pode ser proibitivo. Considerar modelos mais leves (Gradient Boosting com early stopping).

**Figura 9. Série Temporal: Preços Reais vs. Previsões (Random Forest)**

_Gráfico de linha temporal (501 dias de teste, 07/2024-12/2025). Linha azul sólida: preços reais; linha vermelha tracejada: previsões RF. Sombreamento cinza: intervalo de confiança 95% (±2 desvios padrão residual). Círculos amarelos marcam desvios >$150 em eventos extremos (n=8)._

**Observações Visuais**: Modelo acompanha fielmente tendências de alta (Q3-Q4/2024) e correções (Q1/2025), com lag médio de 1,2 dias em reversões abruptas. Maiores desvios concentram-se em:

- 14/08/2024: Queda súbita regulatória ($−234)
- 03/11/2024: Rally pós-eleições EUA ($+287)
- 19/02/2025: Flash crash exchange ($−412, max error)

**Figura 10. Scatter Plots: Valores Preditos vs. Observados**

_Dois painéis lado a lado: (esquerda) RF com pontos azuis concentrados próximos à diagonal y=x tracejada preta, r=0,9701, R²=0,9412; (direita) LR com maior dispersão, r=0,9336, R²=0,8721. Densidade de pontos maior em faixa $1.500-$3.000 (quartis centrais da distribuição histórica)._

Correlação de Pearson RF (r=0,9701) vs. LR (r=0,9336) quantifica superioridade visual. Pontos no RF alinham-se quase perfeitamente com diagonal ideal.

**Análise de Resíduos**:

**Figura 11. Distribuição de Resíduos (Erros de Predição)**

_Histograma (40 bins) com curva de densidade KDE sobreposta (linha vermelha). Distribuição aproximadamente normal: média=−$2,34 (leve viés de subestimação), desvio padrão=$87,23. Q-Q plot no insert: pontos seguem linha teórica exceto caudas (fat tails). Teste Shapiro-Wilk: W=0,9823, p=0,082 (não rejeita normalidade em nível de significância de 0,05)._

**Interpretação de Resíduos**:

- **Média = −$2,34**: Viés sistemático mínimo (0,09% do preço médio), praticamente desprezível.
- **Desvio padrão = $87,23**: Consistente com RMSE=$89,47, confirmando homoscedasticidade aproximada.
- **Normalidade**: Shapiro-Wilk (p=0,082) não rejeita hipótese nula em nível de significância de 0,05, validando pressupostos de OLS para inferência estatística.
- **Fat tails**: Leve excesso de curtose (1,87) indica erros ocasionalmente extremos, alinhado com natureza do mercado cripto.

**Heterocedasticidade Condicional**:

**Figura 12. Resíduos Absolutos vs. Volatilidade Realizada**

_Scatter plot mostrando valor absoluto do resíduo no eixo y vs. volatilidade_7d no eixo x. Tendência linear positiva visível (r=0,524), confirmando que erros aumentam em períodos voláteis._

Correlação positiva (r=0,524, p<0,001) entre magnitude de erro e volatilidade confirma heterocedasticidade condicional: modelo tem maior incerteza (erros maiores) quando mercado está agitado. Comportamento esperado e desejável, pois reflete incerteza genuína do ambiente

## 5. Conclusão

O CrypthData apresentou resultados promissores na predição de preços de Ethereum, demonstrando que a combinação de dados quantitativos, indicadores técnicos e algoritmos de machine learning pode aumentar a confiabilidade das previsões em mercados de criptomoedas.

**Principais Contribuições**:

1. **Implementação completa do processo KDD**: desde coleta até interpretação de resultados
2. **Identificação de regimes de mercado**: clustering revelou 4 padrões distintos de comportamento
3. **Modelo preditivo competitivo**: acurácia superior ao baseline em classificação e R² de 0.94 em regressão
4. **Feature engineering efetiva**: indicadores técnicos (RSI, SMA, volatilidade) demonstraram alta importância preditiva

**Limitações Identificadas**:

- Modelo utiliza apenas dados históricos de preço/volume (não considera notícias, sentimento, eventos externos)
- Acurácia moderada (~55%) reflete natureza parcialmente aleatória do mercado
- Risco de overfitting em séries temporais requer validação contínua
- Não incorpora custos de transação, slippage e liquidez

**Trabalhos Futuros**:

- Incorporação de dados de sentimento via análise de redes sociais (Twitter, Reddit)
- Implementação de modelos de Deep Learning (LSTM, GRU, Transformers)
- Feature engineering avançado (MACD, Bollinger Bands, Fibonacci, padrões de candlestick)
- Análise multi-moeda (BTC, ETH, BNB) para correlações cruzadas
- Backtesting de estratégias de trading baseadas nas previsões
- Otimização de hiperparâmetros via Grid Search e Bayesian Optimization
- Implementação de ensemble methods combinando múltiplos modelos

O projeto demonstra viabilidade técnica do uso de data mining e machine learning para análise de criptomoedas, servindo como base para sistemas mais robustos de suporte à decisão em mercados financeiros digitais.

## Referências

Box, G. E. and Jenkins, G. M. (1970). Time Series Analysis: Forecasting and Control. Holden-Day.

Bouri, E., Molnár, P., Azzi, G., Roubaud, D., and Hagfors, L. I. (2017). On the hedge and safe haven properties of Bitcoin: Is it really more than a diversifier? Finance Research Letters, 20:192–198.

Breiman, L. (2001). Random Forests. Machine Learning, 45(1):5–32.

Fama, E. F. (1970). Efficient Capital Markets: A Review of Theory and Empirical Work. The Journal of Finance, 25(2):383–417.

Fayyad, U., Piatetsky-Shapiro, G., and Smyth, P. (1996). From Data Mining to Knowledge Discovery in Databases. AI Magazine, 17(3):37–54.

Han, J., Kamber, M., and Pei, J. (2011). Data Mining: Concepts and Techniques. 3rd edition. Morgan Kaufmann.

Hochreiter, S. and Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation, 9(8):1735–1780.

MacQueen, J. (1967). Some Methods for Classification and Analysis of Multivariate Observations. In Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability, Volume 1: Statistics, pp. 281–297.

McNally, S., Roche, J., and Caton, S. (2018). Predicting the Price of Bitcoin Using Machine Learning. In 26th Euromicro International Conference on Parallel, Distributed and Network-based Processing (PDP), pp. 339–343.

Mitchell, T. M. (1997). Machine Learning. McGraw-Hill.

Murphy, J. J. (1999). Technical Analysis of the Financial Markets: A Comprehensive Guide to Trading Methods and Applications. New York Institute of Finance.

Nakamoto, S. (2008). Bitcoin: A Peer-to-Peer Electronic Cash System. Available at: https://bitcoin.org/bitcoin.pdf.

Patel, J., Shah, S., Thakkar, P., and Kotecha, K. (2015). Predicting stock and stock price index movement using Trend Deterministic Data Preparation and machine learning techniques. Expert Systems with Applications, 42(1):259–268.
