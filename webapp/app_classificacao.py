#importar bibliotecas
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
import time


#importar arquivos
df = pd.read_csv('dados/dataset_tratado.csv', sep=',')
df_class2 = df[df['Class'] == 2]
df_class4 = df[df['Class'] == 4]

modelo = joblib.load('webapp/modelo.joblib')


#configuração padrão da página
st.set_page_config(page_title='Predict Cancer', layout='centered')


#descrição página principal
st.title('MODELO DE PREVISÃO PARA DETECÇÃO DE CÂNCER DE MAMA')

with st.expander("Descrição"):
    st.write("""
     
No Brasil o câncer de mama é o terceiro tumor com maior incidência,
podendo ter sua evolução de forma mais rápida ou não. Por isso, o 
melhor tratamento começa sempre com um diagnóstico precoce.
Partindo dessa afirmação, podemos hoje criar inteligências artificiais
capazes de gerar diagnósticos cada vez mais precisos para auxiliar a tomada
de decisão dos profissionais da saúde. 

Com o imenso volume de dados hoje 
disponíveis, temos potencial para treinar modelos de aprendizado de máquina 
com a finalidade de analisar pacientes com base em um conjunto de características
e assim indicar a probabilidade de este conter alguma anomalia.

     """)
    st.image("https://static.wixstatic.com/media/nsplsh_49436d2d634549466b3449~mv2.jpg/v1/fill/w_1175,h_550,al_c,q_85,usm_0.66_1.00_0.01,enc_auto/Image%20by%20National%20Cancer%20Institute.jpg")
    

#descrição barra lateral
st.sidebar.caption('# Insira abaixo o diagnóstico de cada característica isolada e, em seguida, faça a previsão:')


#parâmetros do modelo
x = {
    'Clump Thickness': 4.0,
    'Uniformity of Cell Size': 3.0,
    'Uniformity of Cell Shape': 3.0,
    'Marginal Adhesion': 2.0,
    'Single Epithelial Cell Size': 3.0,
    'Bare Nuclei': 3.0,
    'Bland Chromatin': 3.0,
    'Normal Nucleoli': 2.0,
    'Mitoses': 1.0
    }

for item in x:
    valor = st.sidebar.slider(label=f'{item}' ,min_value=1.0, value=x[item], max_value=10.0, step=0.01)
    #receber valores do input    
    x[item] = valor


valores_x = pd.DataFrame(x, index=[0])


#criar botao para rodar modelo
botao = st.sidebar.button('Fazer Previsão')
if botao:
    #fazer previsão
    classificar = modelo.predict(valores_x)
    
    prob_predict = modelo.predict_proba(valores_x)
    class_2 = prob_predict[0][0]
    class_4 = prob_predict[0][1]
    
    #aguardar conclusão da previsão
    with st.sidebar:
        with st.spinner('Aguarde um momento...'):
            time.sleep(1)
    
    #exibir resultado
    if classificar[0] == 2:
        st.sidebar.success(
        '### Tumor classificado como Benigno!\n###### *{:.2%} de chance de não ser maligno.'.format(class_2)
                  )
    else:
        st.sidebar.error(
        '### Tumor classificado como Maligno!\n###### *{:.2%} de chance de ser maligno.'.format(class_4)
                )
        
st.sidebar.markdown("<div> <br> </div>", unsafe_allow_html=True)

#interface gráfica
st.markdown('#### Gráfico de Distribuição de Frequência:')
categoria_grafico = st.selectbox('Selecione a característica a ser visualizada:',
                                 options = ['Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                                            'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei',
                                            'Bland Chromatin', 'Normal Nucleoli', 'Mitoses'])
st.markdown('*características das mulheres monitoradas no grupo de estudo')
fig = go.Figure()
fig.add_trace(go.Histogram(x=df_class2[categoria_grafico], name='Pacientes com Tumor Benigno', marker_color='#40A8C0'))
fig.add_trace(go.Histogram(x=df_class4[categoria_grafico], name='Pacientes com Tumor Maligno', marker_color='#B3093F'))
fig.add_vline(x=x[categoria_grafico], line_width=3, line_dash="dash", line_color="yellow")
fig.update_traces(opacity=0.8)
fig.update_layout(yaxis_title_text='Quantidade', xaxis_title_text=categoria_grafico, barmode='overlay',
                  legend_orientation='h', legend_font_size=14, legend_y=1.13, legend_x=0)

st.plotly_chart(fig, use_container_width=True)
        
    
#rodape
st.write('''*Estes dados foram obtidos dos Hospitais da Universidade de Wisconsin, em Madison-USA.\n
*Responsável pelos dados: Dr. William H. Wolberg.\n
O desenvolvimento deste trabalho pode ser encontrado em: [GitHub](https://github.com/MarioLisboaJr/predict_cancer)''',
         unsafe_allow_html=True)

github = f"<a href='https://github.com/MarioLisboaJr'><img src='https://cdn-icons-png.flaticon.com/512/733/733553.png' height='40' width='40'></a>"
linkedin = f"<a href='https://www.linkedin.com/in/mario-lisboa/'><img src='https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg' height='40' width='40'></a>"
portfolio = f"<a href='https://lisboamario.wixsite.com/website'><img src='https://img.icons8.com/clouds/344/external-link.png' height='50' width='50'></a>"

st.markdown(f"<div style='text-align: center'><br><br><br>{github}&ensp;{linkedin}&ensp;{portfolio}<p>{'Mário Lisbôa'}</div>",
            unsafe_allow_html=True)
