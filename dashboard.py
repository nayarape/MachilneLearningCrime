import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from datetime import datetime, timedelta

# Configuração da página
st.set_page_config(
    page_title='🚓 Crime Analytics Pro', 
    layout='wide',
    initial_sidebar_state='expanded'
)

# CSS customizado para melhorar a aparência
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #FF6B6B;
    }
    .upload-zone {
        border: 2px dashed #4ECDC4;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Header principal
st.markdown("""
<div class="main-header">
    <h1>🚓 Crime Analytics Pro Dashboard</h1>
    <p>Análise Inteligente de Dados Criminais com Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# Instruções sobre onde colocar o arquivo CSV
with st.expander("📋 INSTRUÇÕES: Como usar este dashboard", expanded=True):
    st.markdown("""
    ### 🎯 **Passo a passo:**
    
    1. **Prepare seu arquivo CSV** com dados criminais
    2. **Clique no botão de upload** na barra lateral ⬅️
    3. **Selecione seu arquivo** ou arraste e solte
    4. **Explore as análises** nas diferentes abas
    
    ### 📊 **Colunas esperadas:**
    - `date` - Data do crime
    - `crime_type` - Tipo de crime
    - `location` - Localização
    - `district` - Distrito/Bairro
    - Outras colunas relevantes
    """)

# Sidebar
st.sidebar.markdown("## 📊 Configurações")

uploaded_file = st.sidebar.file_uploader(
    '📁 Upload seu arquivo CSV:',
    type=['csv'],
    help="Selecione um arquivo CSV com dados criminais"
)

if uploaded_file:
    try:
        # Carregamento com indicador de progresso
        with st.spinner('📥 Carregando dados...'):
            df = pd.read_csv(uploaded_file)
        
        st.success(f"✅ {len(df)} registros carregados com sucesso!")
        
        # Processamento automático de datas
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'data' in col.lower()]
        if date_columns:
            date_col = st.sidebar.selectbox("📅 Coluna de data:", date_columns)
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.dropna(subset=[date_col])
        
        # Filtros na sidebar
        st.sidebar.markdown("### 🎛️ Filtros")
        
        # Filtro de crime
        crime_columns = [col for col in df.columns if 'crime' in col.lower() or 'tipo' in col.lower()]
        if crime_columns:
            crime_col = st.sidebar.selectbox("🔍 Coluna de crime:", crime_columns)
            available_crimes = sorted(df[crime_col].dropna().unique())
            selected_crimes = st.sidebar.multiselect(
                'Selecionar tipos de crime:',
                options=available_crimes,
                default=available_crimes[:min(5, len(available_crimes))]
            )
            if selected_crimes:
                df = df[df[crime_col].isin(selected_crimes)]
        
        # Filtro de data
        if date_columns:
            min_date = df[date_col].min().date()
            max_date = df[date_col].max().date()
            
            date_range = st.sidebar.date_input(
                '📅 Período:',
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
            
            if len(date_range) == 2:
                start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
                df = df[(df[date_col] >= start_date) & (df[date_col] <= end_date)]
        
        # Verificar se há dados após filtros
        if not df.empty:
            # Métricas principais
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📊 Total de Registros", f"{len(df):,}")
            
            with col2:
                if crime_columns:
                    st.metric("🏷️ Tipos de Crime", df[crime_col].nunique())
            
            with col3:
                if date_columns:
                    period_days = (df[date_col].max() - df[date_col].min()).days
                    st.metric("📅 Período (dias)", period_days)
            
            with col4:
                if date_columns:
                    daily_avg = len(df) / max(period_days, 1)
                    st.metric("📈 Média Diária", f"{daily_avg:.1f}")
            
            # Tabs principais
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Visão Geral", "📈 Análise Temporal", "🗺️ Análise Espacial", "🤖 Machine Learning"])
            
            with tab1:
                st.subheader("📊 Análise Geral dos Crimes")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if crime_columns:
                        # Top crimes
                        crime_counts = df[crime_col].value_counts().head(10)
                        fig_bar = px.bar(
                            x=crime_counts.values,
                            y=crime_counts.index,
                            orientation='h',
                            title="🏆 Top 10 Tipos de Crime",
                            labels={'x': 'Quantidade', 'y': 'Tipo'},
                            color=crime_counts.values,
                            color_continuous_scale='viridis'
                        )
                        fig_bar.update_layout(height=400, showlegend=False)
                        st.plotly_chart(fig_bar, use_container_width=True)
                
                with col2:
                    if crime_columns:
                        # Pizza chart
                        fig_pie = px.pie(
                            values=crime_counts.values[:8],
                            names=crime_counts.index[:8],
                            title="🥧 Distribuição dos Crimes"
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                
                # Estatísticas descritivas
                st.subheader("📈 Estatísticas Descritivas")
                st.dataframe(df.describe(include='all'), use_container_width=True)
            
            with tab2:
                if date_columns:
                    st.subheader("📈 Análise Temporal")
                    
                    # Crimes por dia
                    df['date_only'] = df[date_col].dt.date
                    daily_crimes = df.groupby('date_only').size().reset_index(name='crimes')
                    daily_crimes['date_only'] = pd.to_datetime(daily_crimes['date_only'])
                    
                    fig_timeline = px.line(
                        daily_crimes,
                        x='date_only',
                        y='crimes',
                        title="📅 Evolução Diária dos Crimes",
                        labels={'crimes': 'Número de Crimes', 'date_only': 'Data'}
                    )
                    fig_timeline.update_traces(line_color='#FF6B6B', line_width=2)
                    st.plotly_chart(fig_timeline, use_container_width=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Por dia da semana
                        df['weekday'] = df[date_col].dt.day_name()
                        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                        weekday_counts = df['weekday'].value_counts().reindex(weekday_order)
                        
                        fig_weekday = px.bar(
                            x=weekday_counts.index,
                            y=weekday_counts.values,
                            title="📊 Crimes por Dia da Semana",
                            color=weekday_counts.values,
                            color_continuous_scale='blues'
                        )
                        st.plotly_chart(fig_weekday, use_container_width=True)
                    
                    with col2:
                        # Por mês
                        df['month'] = df[date_col].dt.strftime('%Y-%m')
                        monthly_counts = df['month'].value_counts().sort_index()
                        
                        fig_monthly = px.bar(
                            x=monthly_counts.index,
                            y=monthly_counts.values,
                            title="📅 Crimes por Mês"
                        )
                        fig_monthly.update_layout(xaxis_tickangle=45)
                        st.plotly_chart(fig_monthly, use_container_width=True)
            
            with tab3:
                st.subheader("🗺️ Análise Geográfica")
                
                # Procurar colunas de localização
                location_cols = [col for col in df.columns if any(term in col.lower() 
                                for term in ['location', 'local', 'distrito', 'bairro', 'district'])]
                
                if location_cols:
                    location_col = st.selectbox("📍 Selecionar coluna de localização:", location_cols)
                    
                    # Top localizações
                    location_counts = df[location_col].value_counts().head(15)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_location_bar = px.bar(
                            x=location_counts.values,
                            y=location_counts.index,
                            orientation='h',
                            title="🏆 Top 15 Localizações",
                            labels={'x': 'Crimes', 'y': 'Localização'}
                        )
                        st.plotly_chart(fig_location_bar, use_container_width=True)
                    
                    with col2:
                        # Treemap
                        fig_treemap = px.treemap(
                            names=location_counts.index[:10],
                            values=location_counts.values[:10],
                            title="🗺️ Mapa de Calor - Localizações"
                        )
                        st.plotly_chart(fig_treemap, use_container_width=True)
                else:
                    st.info("🔍 Nenhuma coluna de localização encontrada nos dados.")
            
            with tab4:
                st.subheader("🤖 Machine Learning")
                
                # Seleção de variáveis
                col1, col2 = st.columns(2)
                
                with col1:
                    target_options = [col for col in df.columns if df[col].dtype in ['object', 'int64', 'float64']]
                    target_column = st.selectbox("🎯 Variável alvo:", target_options)
                
                with col2:
                    test_size = st.slider("📊 % para teste:", 10, 40, 20)
                
                # Configurações do modelo
                with st.expander("⚙️ Configurações Avançadas"):
                    col1, col2 = st.columns(2)
                    with col1:
                        n_estimators = st.slider("🌳 Número de árvores:", 50, 300, 100)
                    with col2:
                        max_depth = st.slider("📏 Profundidade máxima:", 3, 20, 10)
                
                if st.button("🚀 Treinar Modelo", type="primary"):
                    try:
                        with st.spinner("🤖 Treinando modelo..."):
                            # Preparar dados
                            X = df.drop(columns=[target_column])
                            y = df[target_column]
                            
                            # Remover colunas de data
                            date_cols_to_remove = [col for col in X.columns if X[col].dtype == 'datetime64[ns]']
                            if date_cols_to_remove:
                                X = X.drop(columns=date_cols_to_remove)
                            
                            # Identificar tipos de colunas
                            categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
                            numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
                            
                            # Pipeline
                            preprocessor = ColumnTransformer([
                                ('num', StandardScaler(), numerical_cols),
                                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
                            ])
                            
                            model = Pipeline([
                                ('preprocessor', preprocessor),
                                ('classifier', RandomForestClassifier(
                                    n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    random_state=42
                                ))
                            ])
                            
                            # Treino
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=test_size/100, random_state=42
                            )
                            
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            accuracy = accuracy_score(y_test, y_pred)
                            
                            # Resultados
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.success("✅ Modelo treinado!")
                                st.metric("🎯 Acurácia", f"{accuracy:.2%}")
                                
                                with st.expander("📊 Relatório Completo"):
                                    st.text(classification_report(y_test, y_pred))
                            
                            with col2:
                                # Matriz de confusão simples
                                cm = confusion_matrix(y_test, y_pred)
                                fig_cm = px.imshow(
                                    cm,
                                    title="Matriz de Confusão",
                                    color_continuous_scale='Blues',
                                    aspect='auto'
                                )
                                st.plotly_chart(fig_cm, use_container_width=True)
                            
                            # Feature importance
                            if hasattr(model.named_steps['classifier'], 'feature_importances_'):
                                feature_names = model.named_steps['preprocessor'].get_feature_names_out()
                                importances = model.named_steps['classifier'].feature_importances_
                                
                                importance_df = pd.DataFrame({
                                    'feature': feature_names,
                                    'importance': importances
                                }).sort_values('importance', ascending=False).head(10)
                                
                                fig_importance = px.bar(
                                    importance_df,
                                    x='importance',
                                    y='feature',
                                    orientation='h',
                                    title="🏆 Top 10 Features Importantes"
                                )
                                st.plotly_chart(fig_importance, use_container_width=True)
                            
                            # Download do modelo
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            model_name = f'crime_model_{timestamp}.joblib'
                            joblib.dump(model, model_name)
                            
                            with open(model_name, 'rb') as f:
                                st.download_button(
                                    "💾 Download Modelo",
                                    data=f.read(),
                                    file_name=model_name,
                                    mime='application/octet-stream'
                                )
                    
                    except Exception as e:
                        st.error(f"❌ Erro: {str(e)}")
        
        else:
            st.warning("⚠️ Nenhum dado encontrado após aplicar filtros.")
    
    except Exception as e:
        st.error(f"❌ Erro ao carregar arquivo: {str(e)}")

else:
    # Página inicial
    st.markdown("""
    <div class="upload-zone">
        <h2>🚀 Bem-vindo ao Crime Analytics Pro!</h2>
        <p style="font-size: 18px;">Faça upload do seu CSV na barra lateral para começar ⬅️</p>
        <br>
        <h3>✨ Funcionalidades:</h3>
        <ul style="text-align: left; max-width: 500px; margin: 0 auto;">
            <li>📊 Análises estatísticas detalhadas</li>
            <li>📈 Visualizações interativas</li>
            <li>🗺️ Análise geográfica</li>
            <li>🤖 Machine Learning</li>
            <li>💾 Download de modelos</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Dados de exemplo
    st.subheader("📋 Exemplo de formato CSV:")
    example_df = pd.DataFrame({
        'date': ['2024-01-15', '2024-01-16', '2024-01-17'],
        'crime_type': ['Roubo', 'Furto', 'Assalto'],
        'location': ['Centro', 'Shopping', 'Praça'],
        'district': ['Boa Viagem', 'Recife', 'Casa Forte']
    })
    st.dataframe(example_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("🚓 **Crime Analytics Pro** - Análise Inteligente de Dados Criminais")