# Projeto: Transformação Digital de Dados — HealthCare Solutions

**Objetivo:** construir um projeto prático de Ciência de Dados pronto para subir ao GitHub que mostra coleta/geração de dados simulados de saúde, limpeza e pré-processamento, EDA (análise exploratória), modelagem preditiva (risco de readmissão em 30 dias) e uma visualização interativa (Streamlit). Tudo em português e organizado com instruções para reprodução.

---

## Estrutura sugerida do repositório

```
healthcare-solutions-ds/
├── data/
│   ├── raw/                      # CSVs originais gerados
│   │   └── synthetic_patients.csv
│   └── processed/                # CSVs pós-limpeza
│       └── patients_clean.csv
├── notebooks/
│   ├── 01_data_generation.ipynb
│   ├── 02_eda.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── generate_synthetic_data.py
│   ├── clean_and_preprocess.py
│   ├── eda_plots.py
│   ├── train_model.py
│   └── app_streamlit.py
├── requirements.txt
├── README.md
└── .github/
    └── workflows/python-app.yml
```

---

## 1) Coleta / Dados (simulados)

Como os dados reais de hospitais são sensíveis, usamos um *dataset sintético realista* gerado localmente e salvo em `data/raw/synthetic_patients.csv`.

### Esquema do CSV (`synthetic_patients.csv`)

Colunas (explicação rápida):

* `patient_id` (string) — identificador anônimo
* `age` (int)
* `sex` (categorical) — 'M' ou 'F'
* `comorbidities_count` (int) — número de comorbidades
* `chronic_conditions` (list-string) — texto com condições separadas por `;` (ex: "diabetes;hypertension")
* `previous_admissions` (int)
* `length_of_stay` (float) — dias
* `days_since_last_discharge` (int or NA)
* `avg_hr` (float) — frequência cardíaca média durante internação
* `avg_sys_bp` (float) — pressão sistólica média
* `wbc` (float) — contagem de leucócitos
* `lab_cr` (float) — creatinina urinária (exemplo)
* `device_monitored` (0/1) — recebeu monitoramento contínuo
* `satisfaction_score` (0-10) — pesquisa pós-alta
* `readmitted_30d` (0/1) — alvo: readmissão em 30 dias

---

## 2) Script para gerar os dados (src/generate_synthetic_data.py)

```python
# src/generate_synthetic_data.py
import numpy as np
import pandas as pd
from faker import Faker
import random

fake = Faker()
Faker.seed(42)
np.random.seed(42)

N = 5000
rows = []
CHRONIC = ['diabetes','hypertension','copd','asthma','ckd','heart_failure','none']

for i in range(N):
    pid = f"P{100000 + i}"
    age = int(np.clip(np.random.normal(65, 16), 18, 100))
    sex = np.random.choice(['M','F'])
    comorb = np.random.poisson(1.2)
    conds = []
    if comorb == 0:
        conds = ['none']
    else:
        conds = list(np.random.choice(CHRONIC[:-1], size=min(len(CHRONIC)-1, comorb), replace=False))
    prev_adm = np.random.poisson(0.5)
    los = float(np.clip(np.random.exponential(4) + 1, 1, 60))
    days_since = np.nan if np.random.rand() < 0.6 else int(np.random.exponential(60))
    avg_hr = float(np.clip(np.random.normal(78, 12), 40, 160))
    avg_sys_bp = float(np.clip(np.random.normal(130, 18), 80, 220))
    wbc = float(np.clip(np.random.normal(7.5, 3), 1, 30))
    lab_cr = float(np.clip(np.random.normal(1.1, 0.8), 0.3, 10))
    device = int(np.random.rand() < 0.25)
    satisfaction = int(np.clip(np.random.normal(7, 2), 0, 10))

    # risco base para readmissão: função simples
    risk = (0.02 * (age-50) + 0.15*comorb + 0.2*prev_adm + 0.1*(los>7) + 0.1*(lab_cr>1.5) - 0.03*satisfaction)
    p_read = 1/(1+np.exp(- ( -2 + risk )))
    readm = int(np.random.rand() < p_read)

    rows.append({
        'patient_id': pid,
        'age': age,
        'sex': sex,
        'comorbidities_count': comorb,
        'chronic_conditions': ';'.join(conds),
        'previous_admissions': prev_adm,
        'length_of_stay': round(los,1),
        'days_since_last_discharge': days_since,
        'avg_hr': round(avg_hr,1),
        'avg_sys_bp': round(avg_sys_bp,1),
        'wbc': round(wbc,2),
        'lab_cr': round(lab_cr,2),
        'device_monitored': device,
        'satisfaction_score': satisfaction,
        'readmitted_30d': readm
    })

df = pd.DataFrame(rows)

# inserir alguns valores ausentes intencionais e duplicatas
for _ in range(120):
    idx = np.random.choice(df.index)
    col = np.random.choice(['wbc','lab_cr','satisfaction_score','days_since_last_discharge'])
    df.at[idx, col] = np.nan

# duplicar algumas linhas
dups = df.sample(10, random_state=1)
df = pd.concat([df, dups], ignore_index=True)

# salvar
import os
os.makedirs('data/raw', exist_ok=True)

df.to_csv('data/raw/synthetic_patients.csv', index=False)
print('CSV gerado: data/raw/synthetic_patients.csv')
```

> Requisitos: `pip install pandas numpy faker`

---

## 3) Limpeza e pré-processamento (src/clean_and_preprocess.py)

Objetivos: remover duplicatas, tratar valores ausentes, transformar `chronic_conditions` em features (count de condições e dummies simples), normalizar/padronizar variáveis numéricas, codificar variáveis categóricas.

```python
# src/clean_and_preprocess.py
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# carregar
df = pd.read_csv('data/raw/synthetic_patients.csv')

# 1. remover duplicatas (com base em patient_id + features)
df = df.drop_duplicates()

# 2. consertar tipos
# garantir patient_id string
df['patient_id'] = df['patient_id'].astype(str)

# 3. feature engineering: transformar chronic_conditions
# contar condições diferentes (exceto 'none')

def count_conditions(text):
    if pd.isna(text):
        return 0
    parts = [p.strip() for p in str(text).split(';') if p.strip()]
    return 0 if (len(parts)==1 and parts[0].lower()=='none') else len(parts)

df['chronic_count'] = df['chronic_conditions'].apply(count_conditions)

# 4. tratar missing
num_cols = ['days_since_last_discharge','wbc','lab_cr','satisfaction_score']
imp_med = SimpleImputer(strategy='median')
df[num_cols] = imp_med.fit_transform(df[num_cols])

# 5. codificação
# sexo -> binária
df['sex_M'] = (df['sex']=='M').astype(int)

# 6. normalização de colunas numéricas (guardar scaler no pipeline real)
scale_cols = ['age','comorbidities_count','previous_admissions','length_of_stay',
              'days_since_last_discharge','avg_hr','avg_sys_bp','wbc','lab_cr','satisfaction_score','chronic_count']
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[scale_cols])
df_scaled = pd.DataFrame(df_scaled, columns=[c + '_s' for c in scale_cols])

df = pd.concat([df.reset_index(drop=True), df_scaled.reset_index(drop=True)], axis=1)

# 7. salvar
import os
os.makedirs('data/processed', exist_ok=True)
df.to_csv('data/processed/patients_clean.csv', index=False)
print('Limpeza concluída. Arquivo salvo: data/processed/patients_clean.csv')
```

> Observação: para produção, trocar SimpleImputer/StandardScaler por pipeline do scikit-learn e serializar (joblib).

---

## 4) Análise exploratória (notebook: notebooks/02_eda.ipynb)

Sugestões de análises e visualizações (com código de exemplo em células):

* Histogramas de `age`, `length_of_stay`, `satisfaction_score`
* Boxplots de `length_of_stay` por `readmitted_30d`
* Heatmap de correlação entre variáveis numéricas (pearson)
* Taxa de readmissão por faixa etária e por número de comorbidades
* Distribuição de `satisfaction_score` vs `readmitted_30d`

Exemplo (célula):

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/processed/patients_clean.csv')

plt.hist(df['age'], bins=20)
plt.title('Distribuição de idade')
plt.xlabel('idade')
plt.ylabel('contagem')
plt.show()
```

Para cada plot, inclua uma interpretação curta (ex.: "A taxa de readmissão cresce com a idade e com o número de comorbidades; pacientes com satisfação baixa têm maior probabilidade de readmissão").

---

## 5) Modelagem preditiva (src/train_model.py)

Objetivo: prever `readmitted_30d` (classificação binária). Usaremos `RandomForestClassifier` com validação cruzada e relatório de métricas (AUC-ROC, precisão, recall, f1).

```python
# src/train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import joblib

# carregar
df = pd.read_csv('data/processed/patients_clean.csv')

# features (exemplo): usar colunas scaled e sex_M e device_monitored
feature_cols = [c for c in df.columns if c.endswith('_s')] + ['sex_M','device_monitored']
X = df[feature_cols]
y = df['readmitted_30d']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# avaliação
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:,1]
print('AUC-ROC:', roc_auc_score(y_test, y_proba))
print(classification_report(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))

# importância de features
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print('Top features:\n', importances.head(10))

# salvar modelo e colunas
joblib.dump(rf, 'models/rf_readmit.joblib')
joblib.dump(feature_cols, 'models/feature_cols.joblib')
print('Modelo salvo em models/rf_readmit.joblib')
```

**Dica de tuning:** usar `GridSearchCV` sobre `n_estimators`, `max_depth` e `min_samples_leaf`.

---

## 6) Visualização interativa (Streamlit) — `src/app_streamlit.py`

Aplicativo simples que permite: filtro por faixa etária, mostrar taxa de readmissão, gráficos interativos (matplotlib/plotly) e predição por formulário (inserir valores e obter probabilidade de readmissão usando o modelo salvo).

```python
# src/app_streamlit.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(layout='wide', page_title='HealthCare Solutions - Dashboard')

@st.cache_data
def load_data():
    return pd.read_csv('data/processed/patients_clean.csv')

@st.cache_resource
def load_model():
    model = joblib.load('models/rf_readmit.joblib')
    feat_cols = joblib.load('models/feature_cols.joblib')
    return model, feat_cols

df = load_data()
model, feat_cols = load_model()

st.title('Dashboard — Readmissão em 30 dias')

# filtros
age_min, age_max = st.slider('Faixa etária', 18, 100, (30, 90))
sex = st.selectbox('Sexo', ['Ambos','M','F'])

mask = (df['age']>=age_min) & (df['age']<=age_max)
if sex != 'Ambos':
    mask &= (df['sex']==sex)

subset = df[mask]

col1, col2 = st.columns(2)
with col1:
    st.metric('Pacientes no conjunto', len(subset))
    st.metric('Taxa de readmissão', f"{subset['readmitted_30d'].mean():.2%}")

with col2:
    st.bar_chart(subset['readmitted_30d'].value_counts().sort_index())

st.markdown('### Predição rápida (exemplo)')
# formulário simples para predição
with st.form('pred_form'):
    age = st.number_input('Idade', 18, 100, 70)
    comorb = st.number_input('Número de comorbidades', 0, 10, 1)
    prev_adm = st.number_input('Admissões anteriores', 0, 10, 0)
    los = st.number_input('Tempo de permanência (dias)', 1.0, 60.0, 4.0)
    satisfaction = st.number_input('Satisfação (0-10)', 0, 10, 7)
    sex_M = st.selectbox('Sexo', ['M','F']) == 'M'
    device = st.checkbox('Monitoramento contínuo')
    st.form_submit_button('Calcular probabilidade')

    # transformação simplificada: usar média e scaling aproximado
    # Para produção devemos aplicar o mesmo scaler usado no treinamento (serializado).

    # Esta é uma demo simplificada: aplicamos mean-centering usando os dados
    X_mean = df[feat_cols].mean()
    X_std = df[feat_cols].std()

    vals = np.zeros(len(feat_cols))
    # preencher valores correspondentes (nome termina com _s)
    mapping = {
        'age_s': (age - df['age'].mean())/df['age_s'].std(),
        # ... para simplicidade, setar outras features a média
    }
    # versão demo: entrada manual não padronizada -> não usar em ambiente clinico

    st.warning('Predição: esta interface é demo; para produção aplique pipeline idêntico ao de treinamento.')

st.write('---')
st.write('Mais análises e gráficos no notebook `notebooks/02_eda.ipynb`.')
```

> Para rodar o app: `streamlit run src/app_streamlit.py` (executar no venv com dependências instaladas).

---

## 7) Requisitos (`requirements.txt`)

```
pandas
numpy
scikit-learn
matplotlib
seaborn
streamlit
joblib
faker
plotly
```

---

## 8) Boas práticas e considerações éticas

* **Anonimização**: esse projeto usa dados sintéticos. Ao trabalhar com dados reais, REMOVA identificadores diretos e avaliar risco de reidentificação.
* **Avaliação clínica**: modelos que decidem sobre pacientes exigem validação prospectiva e avaliação por especialistas médicos.
* **Viés**: teste performance por subgrupos (idade, sexo, condição crônica) para não introduzir vieses.
* **Governança**: mantenha logs de versões do modelo e dados, e política de privacidade clara.

---

## 9) Sugestão de README (para GitHub)

Inclua no `README.md` do repositório um resumo do projeto, como rodar, exemplos de comandos e a licença (ex: MIT). Um exemplo mínimo:

```markdown
# HealthCare Solutions — Projeto de Ciência de Dados

Resumo: projeto demonstrativo que gera dados sintéticos de hospitais, realiza EDA, treina um modelo de readmissão em 30 dias e publica um dashboard em Streamlit.

## Como rodar (local)

1. `python -m venv .venv` && `source .venv/bin/activate` (ou equivalente no Windows)
2. `pip install -r requirements.txt`
3. `python src/generate_synthetic_data.py`
4. `python src/clean_and_preprocess.py`
5. Abrir `notebooks/02_eda.ipynb` para EDA
6. `python src/train_model.py`
7. `streamlit run src/app_streamlit.py`

## Estrutura de arquivos
Explicação das pastas e arquivos.

## Licença
MIT
```

---

## 10) Checklist para subir ao GitHub

* [ ] Criar `.gitignore` (incluir `/data/processed`, `/models`, `.venv`)
* [ ] Incluir `LICENSE` (MIT)
* [ ] Incluir `README.md` com instruções de execução
* [ ] Subir scripts em `src/`, notebooks em `notebooks/`, dados brutos em `data/raw/` (ou usar Git LFS se grandes)
* [ ] Incluir badges (Python version, build) opcional

---

## 11) Próximos passos sugeridos (se quiser estender)

* Validar modelo com técnicas de interpretação (SHAP) para explicar previsões.
* Implementar pipeline `sklearn` com `ColumnTransformer` e `Pipeline` e serializar `scaler` + `model`.
* Testes unitários para scripts e GitHub Actions para CI.
* Subir um pequeno `Dockerfile` para facilitar deploy do Streamlit.

---

Se quiser, eu posso:

* gerar o repositório completo aqui (arquivos prontos) e empacotar como zip;
* converter os scripts em notebooks executáveis;
* adicionar code cells completas para `notebooks/01_data_generation.ipynb` e `notebooks/02_eda.ipynb`.

Diga qual opção prefere e eu gero os arquivos restantes prontos para download no repositório.
