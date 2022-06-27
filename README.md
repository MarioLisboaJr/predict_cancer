<div style="text-align: center">
    <h1> MODELO DE PREVISÃO PARA DETECÇÃO DE CANCER DE MAMA </h1>
</div>


![](https://static.wixstatic.com/media/nsplsh_49436d2d634549466b3449~mv2.jpg/v1/fill/w_1175,h_550,al_c,q_85,usm_0.66_1.00_0.01,enc_auto/Image%20by%20National%20Cancer%20Institute.jpg)

<br>
<br>


<div style="text-align: left;">
    <a href="https://mariolisboajr-predict-cancer-webappapp-classificacao-6iofe7.streamlitapp.com/"><img src="https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png" width=15%></a>        
</div> 

No Brasil o câncer de mama é o terceiro tumor com maior incidência, podendo ter sua evolução de forma mais rápida ou não. Por isso, o melhor tratamento começa sempre com um diagnóstico precoce. Partindo dessa afirmação, podemos hoje criar inteligências artificiais capazes de gerar diagnósticos cada vez mais precisos para auxiliar a tomada de decisão dos profissionais da saúde.
Com o imenso volume de dados hoje disponíveis, temos potencial para treinar modelos de aprendizado de máquina com a finalidade de analisar pacientes com base em um conjunto de características e assim indicar a probabilidade de este conter alguma anomalia.

<br>
<br>
<hr>

<h2> OBJETIVO </h2>

<br>

**Classificar Tumor na Mama em Benigno ou Maligno**

Desenvolvimento:

- Extrair, Tratar e Limpar Dados
- Visualizar Dados
- Criar Modelo Preditivo
- Disponibilizar Modelo para Utilização

**Desenvolvimento do Projeto**: [Aqui](https://github.com/MarioLisboaJr/predict_cancer/blob/main/notebook/notebook.ipynb)

**Desenvolvimento do Web App**: [Aqui](https://github.com/MarioLisboaJr/predict_cancer/blob/main/webapp/app_classificacao.py)
    
**Aplicativo para Classificação**: [Aqui](https://mariolisboajr-predict-cancer-webappapp-classificacao-6iofe7.streamlitapp.com/)

![GIF](https://github.com/MarioLisboaJr/predict_cancer/blob/main/webapp/webapp.gif?raw=true)
    
<hr>

<h2> INFORMAÇÃO SOBRE OS DADOS </h2>

**Os dados deste problema foi obtido dos Hospitais da Universidade de Wisconsin, em Madison-USA**. <br>
**Responsável: Dr. William H. Wolberg**. <br>

Link para os dados: [Aqui](https://github.com/MarioLisboaJr/predict_cancer/blob/main/dados/breast_cancer_bd.csv) <br>
Mais informações sobre os dados ein: [Descrição](https://github.com/MarioLisboaJr/predict_cancer/blob/main/dados/breast-cancer-wisconsin.names) / [Wisconsin Diagnostic Breast Cancer](https://github.com/MarioLisboaJr/predict_cancer/blob/main/dados/wdbc.names) <br>

A base de dados possui informações sobre 9 exames distintos de 699 mulheres diagnosticadas com tumor maligno ou benigno na mama.

Todos os resultados dos exames são disponibilizados em uma escala de 1 a 10: <br>

1) Clump Thickness <br>
2) Uniformity of Cell Size <br>
3) Uniformity of Cell Shape <br>
4) Marginal Adhesion <br>
5) Single Epithelial Cell Size <br>
6) Bare Nuclei <br>
7) Bland Chromatin <br>
8) Normal Nucleoli <br>
9) Mitoses <br>

<hr>

<h2> VISUALIZAÇÃO DOS DADOS </h2>

<br>

A escolha das características foi minuciosa para a realização de um bom diagnóstico. Abaixo pode-se notar que a maioria delas possuem uma correlação consideravelmente forte com a classificação do tumor e que, um valor mais alto nos resultados de cada exame influencia positivamente na aparição de um tumor maligno.

![Clump Thickness](https://github.com/MarioLisboaJr/predict_cancer/blob/main/outputs/output_20_0.png?raw=true)
![Uniformity of Cell Size](https://github.com/MarioLisboaJr/predict_cancer/blob/main/outputs/output_20_1.png?raw=true)
![Uniformity of Cell Shape](https://github.com/MarioLisboaJr/predict_cancer/blob/main/outputs/output_20_2.png?raw=true)
![Marginal Adhesion](https://github.com/MarioLisboaJr/predict_cancer/blob/main/outputs/output_20_3.png?raw=true)
![Single Epithelial Cell Size](https://github.com/MarioLisboaJr/predict_cancer/blob/main/outputs/output_20_4.png?raw=true)
![Bare Nuclei](https://github.com/MarioLisboaJr/predict_cancer/blob/main/outputs/output_20_5.png?raw=true)
![Bland Chromatin](https://github.com/MarioLisboaJr/predict_cancer/blob/main/outputs/output_20_6.png?raw=true)
![Normal Nucleoli](https://github.com/MarioLisboaJr/predict_cancer/blob/main/outputs/output_20_7.png?raw=true)
![Mitoses](https://github.com/MarioLisboaJr/predict_cancer/blob/main/outputs/output_20_8.png?raw=true)

<br>

![Correlacao](https://github.com/MarioLisboaJr/predict_cancer/blob/main/outputs/output_22_0.png?raw=true)

<hr>

<h2> SUPERVISED LEARNING </h2>

Para criação do modelo foi testado cinco algoritmos diferentes:

- Random Forest
- Suport Vector Machine
- Logistic Regression
- K Neighbor Nearest
- Gradient Booster

De um modo geral todos os modelos tiveram resultados satisfatórios.

Pelo contexto da situação, entendemos que, o problema mais grave em um possível erro de classificação seria a não identificação dos tumores malignos. Este tipo de erro poderia atrasar o início de um tratamento, onde o diagnóstico precoce é de suma importância na cura. Assim, neste caso temos um alto custo dos Falsos Negativos, e uma métrica melhor para esta situação seria o Recall.

Como nossos modelos foram igualmente bons, combinamos dois dos melhores avaliados a fim de equilibrar suas fraquezas individuais usando as probabilidades previstas médias para indicar o rótulo da classe. Selecionamos para isso o **Random Forest e o K Nearest Neighbor**, pois, além de obterem os melhores resultados, são conceitualmente diferentes, o que melhorou ainda mais nosso Recall e Acuracidade.

<hr>

<h2> RESULTADOS DO MODELO </h2>

<br>

![](https://github.com/MarioLisboaJr/predict_cancer/blob/main/outputs/output_33_0.png?raw=true)

<br>
Relatório de Classificação:

        precision    recall  f1-score   support

           2       1.00      1.00      1.00        72
           4       1.00      1.00      1.00        33

    accuracy                           1.00       105

<br>

O modelo combinado entre Random Forest e K Nearest Neighbor acertou todos os resultados nos nossos dados de validação, ainda desconhecidos pelo modelo.
<br>
<br>
Temos aqui então um ótimo modelo de classificação, porém, como se trata de diagnósticos extremamente importantes, é sempre bom avaliar não só o rótulo da classificação, mas também, a probabilidade da ocorrência.
<hr>

<br>

**Desenvolvimento do Projeto**: [Aqui](https://github.com/MarioLisboaJr/predict_cancer/blob/main/notebook/notebook.ipynb)

**Desenvolvimento do Web App no Streamlit**: [Aqui](https://github.com/MarioLisboaJr/predict_cancer/blob/main/webapp/app_classificacao.py)

<div style="text-align: left;">
    <a href="https://mariolisboajr-predict-cancer-webappapp-classificacao-6iofe7.streamlitapp.com/"><img src="https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png" width=15%></a>        
</div> 

<br>

<hr>

<h2> AUTOR </h2>

Mário Lisbôa <br>
Pós-Graduando em Data Science e Analytics - USP [🔗](https://mbauspesalq.com/cursos/mba-em-data-science-e-analytics) <br>

<div style="text-align: left;">
        <br>
        <br>
        <a href="https://www.linkedin.com/in/mario-lisboa/">
        <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" width=15%>
    </a> 
</div>      

