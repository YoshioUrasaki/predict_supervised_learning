# APLICAÇÃO DE MACHINE LEARNING E SENSORIAMENTO REMOTO PARA  APRENDIZADO SUPERVISIONADO

Ferdinando Yoshio Agapito Urasaki
yoshio.urasaki@gmail.com

### Este Projeto tem como objetivo apresentar a aplicação do Aprendizado Supervisionado e Algorítmos de Machine Learning na construção de modelos preditivos visando processos semi-automatizados de classificação com utilização de plataforma e softwares livres gratuitos.

---

O Módulo **predict_supervised_learning.py** foi escrito com o objetivo de viabilizar este processo de automação e é formado por um conjunto de classes e métodos para:

> Executar o **Aprendizado Supervisionado** através de algoritmos de **Machine Learning** com **otimização dos hiperparâmetros** e tratamento dos dados desbalanceados pelo método **RENN - Repeated Edited Nearest Neighbour**, salvando as **métricas** obtidas no formato DataFrame do Pandas e gerando arquivo raster no formato **GeoTiff** resultante da predição do modelo treinado.

Para apresentar de forma prática este Projeto, o arquivo ***predict_supervised_learning.pdf*** descreve os **Materiais** e **Métodos** aplicados no Aprendizado Supervisionado da  Cobertura da Terra utilizando Imagem de Satélite Landsat TM 5, com descrição suscinta da fase de **preparação dos dados** utilizando o **QGIS**, sendo a base teórica para esta **DEMONSTRAÇÃO**.

Para agilizar a execução desta **DEMONSTRAÇÃO** foi gerado um **Recorte** dos dados, com 200 x 200 pixels de dimensão, denominados:

- Classe de Cobertura da Terra: *CLIP_COBERTURA_TERRA.tif*
- Imagem Landsat TM 5: *CLIP_RT_L05_L1TP_219076_20100824_T1_B123457.tif*

podendo-se alterar os valores atribuidos as variáveis **features_path**, **target_path** e **dump_path** para os valores abaixo:

---

| VARIÁVEL      | DATASET COMPLETO                                             | DATASET RECORTE                                              |
| ------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| features_path | os.path.join(base_path, 'base', 'RT_L05_L1TP_219076_20100824_T1_B123457.tif') | os.path.join(base_path, 'base', 'CLIP_RT_L05_L1TP_219076_20100824_T1_B123457.tif') |
| target_path   | os.path.join(base_path, 'base', 'COBERTURA_TERRA.tif')       | os.path.join(base_path, 'base', 'CLIP_COBERTURA_TERRA.tif')  |
| dump_path     | os.path.join(base_path, 'dump')                              | os.path.join(base_path, 'dump_demo')                         |

Também está disponível o arquivo do Jupyter Nootbook **predict_supervised_learning.ipynb** contendo a manipulação do Módulo bem como o desenvolvimento do Projeto utilizando como referência o arquivo   ***predict_supervised_learning.pdf*** .