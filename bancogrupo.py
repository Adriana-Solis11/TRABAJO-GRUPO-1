import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(page_title='Predicción de depósitos bancarios', page_icon='🏦', layout='wide')

@st.cache_resource(show_spinner='Entrenando modelos predictivos...')
def entrenar():
    from pathlib import Path
    ruta = Path(__file__).resolve().parent
    df = pd.read_csv(ruta / 'bank.csv')
    X = df[['age', 'balance', 'duration']]
    y = df['deposit']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    modelos = {
        'Árbol de Decisiones': DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=500, random_state=42),
        'SVM': Pipeline([('scaler', StandardScaler()), ('svm', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42))]),
        'Regresión Logística L1': Pipeline([
    ('scaler', StandardScaler()),
    ('lasso', LogisticRegression(
        penalty='l1',
        solver='liblinear',
        C=1.0,
        class_weight='balanced',
        random_state=42
    ))
]),
    }
    metricas = []
    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        pred = modelo.predict(X_test)
        metricas.append({'Modelo': nombre, 'Accuracy': accuracy_score(y_test, pred), 'Precisión (sí)': precision_score(y_test, pred, pos_label='yes'), 'Recall (sí)': recall_score(y_test, pred, pos_label='yes'), 'F1 (sí)': f1_score(y_test, pred, pos_label='yes')})
    return df, modelos, pd.DataFrame(metricas)

df, modelos, metricas = entrenar()
st.title('🏦 Predicción de contratación de depósitos a plazo')
st.write('Aplicación de Machine Learning basada en el estudio de clientes bancarios.')
pestana1, pestana2 = st.tabs(['🔮 Predicción individual', '📊 Evaluación de modelos'])
with pestana1:
    st.subheader('Datos del cliente')
    st.caption('Ingresa los valores del cliente para estimar si contratará un depósito a plazo.')
    c1, c2, c3 = st.columns(3)
    with c1: edad = st.number_input('Edad (años)', min_value=18, max_value=100, value=41)
    with c2: saldo = st.number_input('Saldo bancario (unidades monetarias del dataset)', value=1500, step=100)
    with c3: duracion = st.number_input('Duración del contacto (segundos)', min_value=0, value=370)
    nombre = st.selectbox('Modelo de predicción', list(modelos), index=2)
    if st.button('Predecir contratación', type='primary'):
        entrada = pd.DataFrame([{'age': edad, 'balance': saldo, 'duration': duracion}])
        modelo = modelos[nombre]
        pred = modelo.predict(entrada)[0]
        indice_si = list(modelo.classes_).index('yes')
        probabilidad = float(modelo.predict_proba(entrada)[0][indice_si])
        st.success('Predicción: SÍ contrataría el depósito' if pred == 'yes' else 'Predicción: NO contrataría el depósito')
        st.metric('Probabilidad estimada de contratación', f'{probabilidad:.1%}')
        st.progress(probabilidad)
        st.caption('Probabilidad estimada por el modelo, no una garantía. La duración del contacto se conoce durante o después de la llamada; no es adecuada para priorizar llamadas antes de realizarlas.')
with pestana2:
    st.subheader('Comparación en el conjunto de prueba (20%)')
    st.dataframe(metricas.style.format({c:'{:.2%}' for c in metricas.columns if c != 'Modelo'}), hide_index=True, use_container_width=True)
    st.bar_chart(metricas.set_index('Modelo')[['Accuracy', 'F1 (sí)']])
    st.caption('Misma partición estratificada 80/20 (random_state=42). Las métricas se calculan al ejecutar la aplicación.')
if __name__ == "__main__":
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    if get_script_run_ctx(suppress_warning=True) is None:
        import subprocess
        import sys
        from pathlib import Path

        subprocess.Popen([
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(Path(__file__).resolve())
        ])