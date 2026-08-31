import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Qualidade do Ar | Regimes & Anomalias",
    layout="wide",
    page_icon="☁️"
)

st.title("☁️ Descoberta de Regimes Operacionais e Anomalias")
st.caption("Disciplina de Aprendizado Não Supervisionado | Metodologia CRISP-DM")

# --- CARREGAMENTO E PRÉ-PROCESSAMENTO ROBUSTO (RELATIVO) ---
@st.cache_data
def run_unsupervised_pipeline(uploaded_file=None, n_clusters=3, contamination=0.01):
    df_raw = None
    
    # 1. Tentativa de carregamento (Upload via interface ou busca de caminhos relativos)
    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file, sep=';', decimal=',')
    else:
        root_dir = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
        candidate_paths = [
            root_dir / "data" / "raw" / "AirQualityUCI_export.csv",
            root_dir / "data" / "raw" / "AirQualityUCI.csv",
            Path("data/raw/AirQualityUCI_export.csv"),
            Path("data/raw/AirQualityUCI.csv"),
            Path("AirQualityUCI_export.csv"),
            Path("AirQualityUCI.csv")
        ]
        for p in candidate_paths:
            if p.exists():
                try:
                    df_raw = pd.read_csv(p)
                    break
                except Exception:
                    pass

    if df_raw is None:
        return None

    # Reconstrução temporal (Data/Hora)
    headers = list(df_raw.columns)
    datetimes = None
    if len(headers) >= 2:
        try:
            excel_epoch = datetime(1899, 12, 30)
            datetimes = [
                excel_epoch + timedelta(days=float(r[0]) + float(r[1]))
                for r in df_raw.iloc[:, :2].values
            ]
        except Exception:
            datetimes = pd.to_datetime(df_raw.iloc[:, 0].astype(str) + ' ' + df_raw.iloc[:, 1].astype(str), errors='coerce')
    
    # Tratamento das variáveis analíticas e conversão de -200 para NaN
    feature_cols = [c for c in headers if c not in ['Date', 'Time', 'Unnamed: 0']]
    X_raw = df_raw[feature_cols].apply(pd.to_numeric, errors='coerce').values
    X_raw = X_raw.copy()
    X_raw[X_raw == -200] = np.nan

    # Regra 1: Descarte de variáveis com > 50% de nulos (ex: NMHC(GT))
    missing_rate_col = np.isnan(X_raw).mean(axis=0)
    keep_cols_mask = missing_rate_col <= 0.50
    feature_names = np.array(feature_cols)[keep_cols_mask]
    dropped_features = np.array(feature_cols)[~keep_cols_mask]
    X_filtered = X_raw[:, keep_cols_mask]

    # Regra 2: Identificação de linhas de baixa qualidade (> 30% ausentes)
    row_missing_rate = np.isnan(X_filtered).mean(axis=1)
    low_quality_mask = row_missing_rate > 0.30

    # Imputação: Interpolação curta (<= 3h) + KNN (k=5)
    X_imputed = X_filtered.copy()
    for j in range(X_imputed.shape[1]):
        s = pd.Series(X_imputed[:, j])
        s = s.interpolate(method='linear', limit=3)
        X_imputed[:, j] = s.values

    if np.isnan(X_imputed).any():
        knn = KNNImputer(n_neighbors=5, weights='distance')
        X_imputed = knn.fit_transform(X_imputed)

    # Padronização Robusta (RobustScaler) e Clipping
    scaler = RobustScaler()
    Z = scaler.fit_transform(X_imputed)
    lo, hi = np.percentile(Z, 0.5, axis=0), np.percentile(Z, 99.5, axis=0)
    Z_clipped = np.clip(Z, lo, hi)

    # Agrupamento de Macro-Regimes (K-Means)
    kmeans = KMeans(n_clusters=n_clusters, n_init=30, random_state=42)
    raw_labels = kmeans.fit_predict(Z_clipped)
    
    # Ordenação dos regimes por intensidade mediana de poluentes
    cluster_intensity = {c: float(np.median(Z[raw_labels == c])) for c in range(n_clusters)}
    ordered_raw = sorted(cluster_intensity, key=cluster_intensity.get)
    label_map = {raw: new + 1 for new, raw in enumerate(ordered_raw)}
    regimes = np.array([label_map[x] for x in raw_labels])

    # Redução Dimensional PCA (Visualização 2D)
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(Z_clipped)

    # Detecção de Anomalias Condicionada ao Regime (Isolation Forest por Regime)
    anomaly_score = np.full(len(Z), np.nan)
    anomaly_flag = np.zeros(len(Z), dtype=bool)
    
    for r in np.unique(regimes):
        idx = np.where(regimes == r)[0]
        train_idx = idx[~low_quality_mask[idx]] # Treina apenas com dados de boa qualidade
        if len(train_idx) > 10:
            detector = IsolationForest(n_estimators=200, contamination='auto', random_state=42, n_jobs=-1)
            detector.fit(Z_clipped[train_idx])
            scores = -detector.score_samples(Z_clipped[idx])
            scores_train = -detector.score_samples(Z_clipped[train_idx])
            thresh = np.quantile(scores_train, 1.0 - contamination)
            anomaly_score[idx] = scores
            anomaly_flag[idx] = scores >= thresh

    # Garante que linhas de baixa qualidade não sejam falso-positivos de anomalia
    anomaly_flag = anomaly_flag & ~low_quality_mask

    # Construção do DataFrame Final
    df_res = pd.DataFrame(X_imputed, columns=feature_names)
    if datetimes is not None:
        df_res['DataHora'] = datetimes
    df_res['Regime'] = regimes
    df_res['PCA1'] = pca_coords[:, 0]
    df_res['PCA2'] = pca_coords[:, 1]
    df_res['Score_Anomalia'] = anomaly_score
    df_res['Anomalia'] = anomaly_flag
    df_res['Baixa_Qualidade'] = low_quality_mask

    return {
        'df': df_res,
        'feature_names': feature_names,
        'dropped_features': dropped_features,
        'pca_var': pca.explained_variance_ratio_ * 100
    }

# --- BARRA LATERAL (CONTROLES INTERATIVOS) ---
st.sidebar.header("⚙️ Configurações do Modelo")
uploaded_file = st.sidebar.file_uploader("Upload de CSV \n(O B R I G A T Ó R I O)", type=["csv"])
k_clusters = st.sidebar.slider("Número de Macro-Regimes (K)", min_value=2, max_value=6, value=3)
contamination = st.sidebar.slider("Sensibilidade de Anomalia (%)", min_value=0.5, max_value=5.0, value=1.0) / 100.0

# Execução do Pipeline
if uploaded_file is not None:
    data = run_unsupervised_pipeline(uploaded_file, n_clusters=k_clusters, contamination=contamination)

    if data is None:
        st.error("⚠️ Não foi possível processar o arquivo carregado. Verifique a formatação do arquivo CSV.")
        st.stop()

    df = data['df']
    feats = data['feature_names']

    # --- DASHBOARD: CARDS DE METRICAS ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total de Horas Analisadas", f"{len(df):,}")
    col2.metric("Regimes Identificados", f"{k_clusters}")
    col3.metric("Anomalias Detectadas", f"{df['Anomalia'].sum():,} ({100*df['Anomalia'].mean():.2f}%)")
    col4.metric("Dados Ruins (>30% Nulos)", f"{df['Baixa_Qualidade'].sum():,} ({100*df['Baixa_Qualidade'].mean():.2f}%)")

    st.markdown("---")

    # --- ABAS DE APRESENTAÇÃO ---
    tab1, tab2, tab3 = st.tabs([
        "🧩 Regimes Operacionais (K-Means)", 
        "🚨 Anomalias Condicionadas (Isolation Forest)", 
        "📊 Perfil dos Regimes & Conclusões"
    ])

    with tab1:
        st.subheader("Visualização dos Regimes no Espaço PCA")
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Regime', palette='viridis', alpha=0.6, s=18, ax=ax)
            ax.set_title("Separação dos Clusters (K-Means + PCA)")
            ax.set_xlabel(f"PC1 ({data['pca_var'][0]:.1f}% da variância)")
            ax.set_ylabel(f"PC2 ({data['pca_var'][1]:.1f}% da variância)")
            st.pyplot(fig)
            
        with col_right:
            st.markdown("**Distribuição das Horas por Regime**")
            dist = df['Regime'].value_counts(normalize=True).rename("Proporção").to_frame()
            dist['Horas'] = df['Regime'].value_counts()
            dist['Proporção'] = dist['Proporção'].apply(lambda x: f"{100*x:.1f}%")
            st.dataframe(dist[['Horas', 'Proporção']])

    with tab2:
        st.subheader("Anomalias Detectadas DENTRO de Cada Regime")
        st.caption("Princípio central: Um ponto só é anômalo se for atípico em relação aos pares do seu próprio regime.")
        
        fig2, ax2 = plt.subplots(figsize=(9, 4.5))
        sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Regime', palette='cividis', alpha=0.3, s=15, ax=ax2)
        anom = df[df['Anomalia']]
        ax2.scatter(anom['PCA1'], anom['PCA2'], color='red', marker='x', s=50, label='Anomalia Operacional', zorder=5)
        ax2.legend()
        st.pyplot(fig2)

        st.markdown("### Top Ocorrências de Anomalias")
        cols_view = (['DataHora'] if 'DataHora' in df.columns else []) + ['Regime', 'Score_Anomalia'] + list(feats[:6])
        st.dataframe(df[df['Anomalia']].sort_values(by='Score_Anomalia', ascending=False)[cols_view].head(20))

    with tab3:
        st.subheader("Perfil Mediano dos Regimes (Unidades Originais)")
        profile = df.groupby('Regime')[list(feats)].median()
        st.dataframe(profile.style.highlight_max(axis=0, color='#ffcdd2'))

        st.info(f"💡 **Tratamento de Qualidade**: Variável descartada por ausência excessiva (>50%): `{', '.join(data['dropped_features'])}`.")
        
        st.markdown("""
        ### 📝 Resumo Executivo para a Apresentação:
        1. **Regimes Não Supervisionados**: O K-Means agrupou o histórico em estados operacionais distintos (ex: Madrugada/Baixo Tráfego vs. Horário de Pico).
        2. **Falso Positivo Eliminado**: Linhas com falhas de leitora (>30% de `-200`) foram sinalizadas como **Baixa Qualidade**, impedindo que gerem alarmes operacionais falsos.
        3. **Anomalias Condicionadas**: O Isolation Forest treinado individualmente por regime garante que aumentos esperados de poluição em horários de pico não sejam confundidos com falhas técnicas.
        """)

else:
    st.info("💡 **Aguardando arquivo**: Faça o upload do arquivo `AirQualityUCI.csv` na barra lateral para abrir a análise.")