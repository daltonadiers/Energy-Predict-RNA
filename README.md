# Previsão de Demanda de Eletricidade com Redes Neurais

Este repositório contém o código para um modelo de rede neural que prevê a demanda diária de eletricidade com base em diversas características, incluindo preços de varejo recomendados (RRP), temperatura, exposição solar, precipitação, e informações sobre dias escolares e feriados.

## Estrutura do Projeto

- `complete_dataset.csv`: Arquivo de dados com as informações utilizadas para treinar e testar o modelo.
- `predict_demand.py`: Script principal para treinamento, avaliação e predição utilizando o modelo de rede neural.
- `README.md`: Documento de instruções e informações sobre o projeto.

## Dataset

    O Dataset utilizado "Daily Electricity Price and Demand Data" está disponível no Kaggle, através do link: https://www.kaggle.com/datasets/aramacus/electricity-demand-in-victoria-australia

## Requisitos

Antes de executar o código, certifique-se de ter os seguintes requisitos instalados:

- Python 3.6+
- Bibliotecas Python:
  - pandas
  - numpy
  - scikit-learn
  - tensorflow
  - CUDA Toolkit (se estiver utilizando GPU)
  - cuDNN (se estiver utilizando GPU)

## Instalação

1. Clone o repositório:
   ```bash
   git clone https://github.com/daltonadiers/Energy-Predict-RNA.git
   cd Energy-Predict-RNA
2. Instale as dependências:
    ```bash
    pip install -r requirements.txt
3. Execute o script:
    ```bash
    python main.py

## Sobre

    Esse projeto foi construído para a disciplina de Inteligência Artifical do curso de Ciência da Computação da Universidade de Passo Fundo.

## Autores

    Dalton Oberdan Adiers @daltonadiers
    Miguel Giacomolli Righi

    Você pode entrar em contato através do e-mail: dalton@upf.br