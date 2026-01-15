# 🏅 Dashboard Olímpico Interativo

Este projeto consiste num **dashboard interativo desenvolvido em Streamlit** para análise dos Jogos Olímpicos modernos.  
O objetivo é explorar estatisticamente a participação dos atletas, distribuição de medalhas e evolução das modalidades olímpicas ao longo dos anos.

O projeto foi desenvolvido no âmbito da disciplina **Data Analysis Lab**.

---

## 🎯 Funcionalidades

- Filtros interativos por país, ano, desporto e sexo
- Estatísticas descritivas dos atletas (idade, altura, peso)
- Análises de medalhas por país e por edição olímpica
- Visualizações interativas com Plotly
- Mapa choropleth de medalhas por país
- Normalização e padronização de variáveis
- Análise de Correlação
- Redução de dimensionalidade com PCA
- Modelos de Machine Learning para previsão de medalhas
- Previsão interativa a partir de dados introduzidos pelo utilizador

---

## 📂 Dataset

O projeto utiliza o dataset **Olympic Athletes Dataset** (`athlete_events.csv`), disponível no Kaggle:

https://www.kaggle.com/datasets/heesoo37/120-years-of-olympic-history-athletes-and-results

---

## 🛠️ Tecnologias utilizadas

- Python 3.x  
- Streamlit  
- Pandas  
- NumPy  
- Plotly  
- Matplotlib  
- Seaborn  
- Scikit-learn  

---

## ⚙️ Instalação

### 1. Clonar o repositório

```bash
git clone https://github.com/teu-username/dashboard_olimpico.git
cd dashboard_olimpico
```
2. Instalar dependências
pip install streamlit pandas numpy plotly scikit-learn matplotlib seaborn

▶️ Executar o dashboard

No terminal, dentro da pasta do projeto:

python -m streamlit run dashboard_olimpico.py


Após executar, o Streamlit abrirá automaticamente no navegador:

http://localhost:8501

📊 Estrutura do projeto
dashboard_olimpico/
│
├── dashboard_olimpico.py      # Código principal do dashboard
├── athlete_events.csv         # Dataset
├── README.md                  # Este ficheiro

👤 Autor

Igor Oliveira

Projeto desenvolvido para fins académicos.

📄 Licença

Este projeto é de uso académico e educacional.
