# Teste Data Science - Cognitivo.ai

Os códigos contidos nesse repositório caracterizam a análise exploratória dos dados encontrados em [AirBnb Data](http://insideairbnb.com/get-the-data.html), avaliando sua consistência e correlação.

## Modelagem

Após uma breve verificação das 3 possíveis variáveis resposta, escolhi a segmentação dos principais assuntos das reviews (review_scores_rating).
Entro em mais detalhes nessa tomada de decisão no PPT contido nesse repositório.

## Linguagem

Utilizei Python com Pandas, seaborn, sklearn e numpy para desenvolvimento tanto das visualizações quanto códigos. 
Logo no início dos códigos dá para ver minhas instruções de importação.

```
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,  mean_absolute_error
from sklearn.model_selection import GridSearchCV
import numpy as np
```

## Questionário

Aqui tratarei com o questionário da Cognitivo.ai

### A- Como foi a deifinição da sua estratégia de modelagem?

Usei como base a metodologia de Data Science proposta pora John Rollins em seu [artigo](https://www.ibmbigdatahub.com/blog/why-we-need-methodology-data-science)

Com isso, segui estes passos:

### Passo-a-Passo:
1. A compreensão do Negócio
2. Uma aproximação Analítica dos Dados
3. Requisitos dos dados para cada variável resposta
4. Coleta, Compreensão e Preparo desses dados
5. Ciclos de Modelagem até chegar as conclusões esperadas

### B - Como foi definida a função de Custo utilizada?

Utilizei vários métodos de cálculo da função de custo, em conjunto de indicadores para definir sua qualidade (MAE, MSE, RMSE).

### Os métodos utilizaram para Regressão foram:

1. Decision Tree
2. Bayesian
3. Random Forest
4. Polinomial
5. Logística
6. SVM
7. Rede Neural

Todos foram utilizados no dataframe que criei para review_scores_rating. O qual se mostrou com maior correlação entre suas colunas após tratamento. 

### Qual foi o critério utilizado na seleção do Modelo final?

Após escolher que trabalharia com review_scores_rating, e tratar prontamente seus dados e remover nans. Fiz vários testes executando vários modelos e
utilizando varios metodos de avalicao (MAE, MSE, RMSE). 

O que obteve melhor MSE e maior score (0.71) foi o Random Forest, entretanto, a Rede Neural foi superior de uma forma geral e este foi escolhido.


### D - Qual foi o critério utilizado para validação do Modelo

Utilizei validação cruzada k-fold Cross Validation e calculei a média e desvio padrão dos 10 testes realizado.

### E - Quais evidências você possui de que seu modelo é suficientemente bom?

Além do fato de que os critérios (MAE, MSE, RMSE) e o Cross Validation estão apresentando bons valores, a precisão de correlação de vários componentes do Data Frame ser superior a 50% e apresentar até valores maiores de 70% mostram que poderíamos trabalhar mais a fundo com estes dados.

## Slide com mais detalhes
O slide 'Apresentação-Nielsen.pptx' apresenta mais informações e imagens sobre a implementação e trabalhos futuros.






