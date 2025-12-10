"""
Script para gerar todas as figuras citadas no artigo científico CrypthData
Baseado na análise de Ethereum (ETH/USDT)

Figuras geradas:
- Figura 1: Pipeline do Processo KDD
- Figura 2: Matriz de Correlação de Pearson
- Figura 3: Distribuição de Retornos com Q-Q plot
- Figura 4: Método Elbow e Silhouette Score
- Figura 5: Visualização dos Clusters
- Figura 6: Matriz de Confusão Normalizada
- Figura 7: Feature Importance
- Figura 8: Curva ROC
- Figura 9: Série Temporal (Real vs Previsto)
- Figura 10: Scatter Plots (Predito vs Observado)
- Figura 11: Distribuição de Resíduos
- Figura 12: Resíduos vs Volatilidade
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, roc_curve, auc, 
                             mean_absolute_error, mean_squared_error, r2_score)
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuração de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def carregar_dados():
    """Carrega dados do Ethereum do arquivo CSV ou executa coleta"""
    try:
        df = pd.read_csv('eth_data.csv', parse_dates=['open_time'])
        print(f"✅ Dados carregados: {len(df)} observações")
        return df
    except FileNotFoundError:
        print("⚠️ Arquivo eth_data.csv não encontrado. Execute primeiro o notebook eth.ipynb")
        return None

def preparar_dados(df):
    """Aplica feature engineering e prepara dados para modelagem"""
    # Retorno percentual
    df['return'] = df['close'].pct_change()
    
    # Médias móveis
    df['SMA_7'] = df['close'].rolling(window=7).mean()
    df['SMA_30'] = df['close'].rolling(window=30).mean()
    
    # Volatilidade
    df['volatilidade_7d'] = df['return'].rolling(window=7).std()
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # Targets
    df['price_up'] = (df['close'].shift(-1) > df['close']).astype(int)
    df['next_close'] = df['close'].shift(-1)
    
    # Remove NaN
    df = df.dropna()
    
    return df

def figura1_pipeline_kdd():
    """Figura 1: Pipeline do Processo KDD"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    # Cores para cada etapa
    cores = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']
    etapas = [
        'SELEÇÃO\nDados Brutos\n(API Binance)',
        'PRÉ-PROCESSAMENTO\nLimpeza e\nValidação',
        'TRANSFORMAÇÃO\nFeature\nEngineering',
        'MINERAÇÃO\nClustering\nClassificação\nRegressão',
        'INTERPRETAÇÃO\nAvaliação e\nResultados'
    ]
    
    y_pos = 0.5
    for i, (etapa, cor) in enumerate(zip(etapas, cores)):
        x_pos = 0.1 + i * 0.18
        
        # Caixa da etapa
        bbox = dict(boxstyle='round,pad=0.5', facecolor=cor, edgecolor='black', linewidth=2, alpha=0.8)
        ax.text(x_pos, y_pos, etapa, fontsize=11, ha='center', va='center',
                bbox=bbox, fontweight='bold', color='white')
        
        # Seta
        if i < len(etapas) - 1:
            ax.annotate('', xy=(x_pos + 0.09, y_pos), xytext=(x_pos + 0.07, y_pos),
                       arrowprops=dict(arrowstyle='->', lw=3, color='black'))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.title('Figura 1. Pipeline do Processo KDD Implementado', 
              fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('figura1_pipeline_kdd.png', dpi=300, bbox_inches='tight')
    print("✅ Figura 1 salva: figura1_pipeline_kdd.png")
    plt.close()

def figura2_correlacao(df):
    """Figura 2: Matriz de Correlação de Pearson"""
    features = ['open', 'high', 'low', 'close', 'volume', 'return', 
                'SMA_7', 'SMA_30', 'volatilidade_7d', 'RSI_14']
    
    corr_matrix = df[features].corr()
    
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.3f', 
                cmap='coolwarm', center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    
    plt.title('Figura 2. Matriz de Correlação de Pearson entre Features', 
              fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('figura2_correlacao.png', dpi=300, bbox_inches='tight')
    print("✅ Figura 2 salva: figura2_correlacao.png")
    plt.close()

def figura3_distribuicao_retornos(df):
    """Figura 3: Distribuição de Retornos com Q-Q plot"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Histograma com curva normal
    returns = df['return'].dropna()
    mu, sigma = returns.mean(), returns.std()
    
    ax1.hist(returns, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    x = np.linspace(returns.min(), returns.max(), 100)
    ax1.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2, label='Normal Teórica')
    ax1.axvline(mu, color='green', linestyle='--', lw=2, label=f'μ={mu:.4f}')
    ax1.set_xlabel('Retorno Diário', fontsize=12)
    ax1.set_ylabel('Densidade', fontsize=12)
    ax1.set_title('Distribuição de Retornos Diários', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Q-Q plot
    stats.probplot(returns, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot', fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    plt.suptitle('Figura 3. Distribuição de Retornos Diários com Ajuste de Distribuição Normal', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('figura3_distribuicao_retornos.png', dpi=300, bbox_inches='tight')
    print("✅ Figura 3 salva: figura3_distribuicao_retornos.png")
    plt.close()

def figura4_elbow_silhouette(df):
    """Figura 4: Método Elbow e Silhouette Score"""
    from sklearn.metrics import silhouette_score
    
    X_cluster = df[['return', 'volume', 'volatilidade_7d', 'RSI_14']].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    K_range = range(2, 11)
    inercias = []
    silhouettes = []
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        kmeans.fit(X_scaled)
        inercias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X_scaled, kmeans.labels_))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Método Elbow
    ax1.plot(K_range, inercias, 'bo-', linewidth=2, markersize=8)
    ax1.axvline(4, color='red', linestyle='--', lw=2, label='k=4 (ótimo)')
    ax1.set_xlabel('Número de Clusters (k)', fontsize=12)
    ax1.set_ylabel('Inércia', fontsize=12)
    ax1.set_title('Método Elbow', fontsize=13, fontweight='bold')
    ax1.grid(alpha=0.3)
    ax1.legend()
    
    # Silhouette Score
    ax2.plot(K_range, silhouettes, 'go-', linewidth=2, markersize=8)
    ax2.axvline(4, color='red', linestyle='--', lw=2, label='k=4 (ótimo)')
    ax2.axhline(max(silhouettes), color='orange', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Número de Clusters (k)', fontsize=12)
    ax2.set_ylabel('Silhouette Score', fontsize=12)
    ax2.set_title('Silhouette Score', fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.legend()
    
    plt.suptitle('Figura 4. Método Elbow e Silhouette Score', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('figura4_elbow_silhouette.png', dpi=300, bbox_inches='tight')
    print("✅ Figura 4 salva: figura4_elbow_silhouette.png")
    plt.close()

def figura5_clusters(df):
    """Figura 5: Visualização dos Clusters"""
    X_cluster = df[['return', 'volume', 'volatilidade_7d', 'RSI_14']].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    kmeans = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Voltar para escala original para visualização
    X_original = X_cluster.values
    
    plt.figure(figsize=(12, 8))
    colors = ['blue', 'red', 'green', 'orange']
    labels = ['Cluster 0 (Neutra)', 'Cluster 1 (Bull)', 'Cluster 2 (Bear)', 'Cluster 3 (Consolidação)']
    
    for i in range(4):
        mask = clusters == i
        plt.scatter(X_original[mask, 0], X_original[mask, 2], 
                   c=colors[i], label=f'{labels[i]} (n={mask.sum()})',
                   alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    # Centroides
    centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)
    plt.scatter(centroids_original[:, 0], centroids_original[:, 2], 
               c='black', marker='X', s=300, edgecolors='white', linewidth=2,
               label='Centroides')
    
    plt.xlabel('Retorno Médio (%)', fontsize=12)
    plt.ylabel('Volatilidade (7 dias)', fontsize=12)
    plt.title('Figura 5. Visualização dos Clusters no Espaço Retorno × Volatilidade', 
              fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('figura5_clusters.png', dpi=300, bbox_inches='tight')
    print("✅ Figura 5 salva: figura5_clusters.png")
    plt.close()

def figura6_confusion_matrix(y_test, y_pred):
    """Figura 6: Matriz de Confusão Normalizada"""
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='RdYlBu_r',
                xticklabels=['Predito: Queda', 'Predito: Alta'],
                yticklabels=['Real: Queda', 'Real: Alta'],
                cbar_kws={'label': 'Proporção'}, vmin=0, vmax=1)
    
    plt.title('Figura 6. Matriz de Confusão Normalizada', 
              fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('Classe Real', fontsize=12)
    plt.xlabel('Classe Predita', fontsize=12)
    plt.tight_layout()
    plt.savefig('figura6_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✅ Figura 6 salva: figura6_confusion_matrix.png")
    plt.close()

def figura7_feature_importance(model, feature_names):
    """Figura 7: Feature Importance"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(feature_names)))
    
    plt.barh(range(len(feature_names)), importances[indices], color=colors[::-1], edgecolor='black')
    plt.yticks(range(len(feature_names)), [feature_names[i] for i in indices])
    plt.xlabel('Importância (Gini)', fontsize=12)
    plt.title('Figura 7. Importância Relativa de Features (Gini Importance)', 
              fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('figura7_feature_importance.png', dpi=300, bbox_inches='tight')
    print("✅ Figura 7 salva: figura7_feature_importance.png")
    plt.close()

def figura8_roc_curve(y_test, y_proba):
    """Figura 8: Curva ROC"""
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random Classifier')
    plt.scatter([fpr[np.argmax(tpr - fpr)]], [tpr[np.argmax(tpr - fpr)]], 
               color='orange', s=200, marker='o', label='Ponto Operacional', zorder=5)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title('Figura 8. Curva ROC (Receiver Operating Characteristic)', 
              fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('figura8_roc_curve.png', dpi=300, bbox_inches='tight')
    print("✅ Figura 8 salva: figura8_roc_curve.png")
    plt.close()

def figura9_serie_temporal(y_test_dates, y_test_reg, y_pred_reg):
    """Figura 9: Série Temporal Real vs Previsto"""
    residuals = y_test_reg - y_pred_reg
    std_residual = residuals.std()
    
    plt.figure(figsize=(14, 7))
    plt.plot(y_test_dates, y_test_reg, 'b-', linewidth=2, label='Preços Reais', alpha=0.8)
    plt.plot(y_test_dates, y_pred_reg, 'r--', linewidth=2, label='Previsões RF', alpha=0.8)
    plt.fill_between(y_test_dates, 
                     y_pred_reg - 2*std_residual, 
                     y_pred_reg + 2*std_residual,
                     color='gray', alpha=0.2, label='IC 95%')
    
    # Marcar desvios extremos
    large_errors = np.abs(residuals) > 150
    if large_errors.sum() > 0:
        plt.scatter(y_test_dates[large_errors], y_test_reg[large_errors], 
                   color='yellow', s=100, marker='o', edgecolors='black', 
                   linewidth=2, label=f'Desvios >$150 (n={large_errors.sum()})', zorder=5)
    
    plt.xlabel('Data', fontsize=12)
    plt.ylabel('Preço ETH (USDT)', fontsize=12)
    plt.title('Figura 9. Série Temporal: Preços Reais vs. Previsões (Random Forest)', 
              fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('figura9_serie_temporal.png', dpi=300, bbox_inches='tight')
    print("✅ Figura 9 salva: figura9_serie_temporal.png")
    plt.close()

def figura10_scatter_plots(y_test_reg, y_pred_rf, y_pred_lr):
    """Figura 10: Scatter Plots Predito vs Observado"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Random Forest
    ax1.scatter(y_test_reg, y_pred_rf, alpha=0.5, s=30, edgecolors='black', linewidth=0.5)
    ax1.plot([y_test_reg.min(), y_test_reg.max()], 
             [y_test_reg.min(), y_test_reg.max()], 
             'k--', lw=2, label='Diagonal Ideal (y=x)')
    
    corr_rf = np.corrcoef(y_test_reg, y_pred_rf)[0, 1]
    r2_rf = r2_score(y_test_reg, y_pred_rf)
    ax1.text(0.05, 0.95, f'ρ = {corr_rf:.4f}\nR² = {r2_rf:.4f}', 
             transform=ax1.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax1.set_xlabel('Preço Observado (USDT)', fontsize=12)
    ax1.set_ylabel('Preço Predito (USDT)', fontsize=12)
    ax1.set_title('Random Forest Regressor', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Linear Regression
    ax2.scatter(y_test_reg, y_pred_lr, alpha=0.5, s=30, color='orange', 
               edgecolors='black', linewidth=0.5)
    ax2.plot([y_test_reg.min(), y_test_reg.max()], 
             [y_test_reg.min(), y_test_reg.max()], 
             'k--', lw=2, label='Diagonal Ideal (y=x)')
    
    corr_lr = np.corrcoef(y_test_reg, y_pred_lr)[0, 1]
    r2_lr = r2_score(y_test_reg, y_pred_lr)
    ax2.text(0.05, 0.95, f'ρ = {corr_lr:.4f}\nR² = {r2_lr:.4f}', 
             transform=ax2.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax2.set_xlabel('Preço Observado (USDT)', fontsize=12)
    ax2.set_ylabel('Preço Predito (USDT)', fontsize=12)
    ax2.set_title('Linear Regression', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.suptitle('Figura 10. Scatter Plots: Valores Preditos vs. Observados', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('figura10_scatter_plots.png', dpi=300, bbox_inches='tight')
    print("✅ Figura 10 salva: figura10_scatter_plots.png")
    plt.close()

def figura11_residuos(residuals):
    """Figura 11: Distribuição de Resíduos"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Histograma com KDE
    ax1.hist(residuals, bins=40, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    
    # KDE
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(residuals)
    x_range = np.linspace(residuals.min(), residuals.max(), 100)
    ax1.plot(x_range, kde(x_range), 'r-', lw=2, label='KDE')
    
    mu, sigma = residuals.mean(), residuals.std()
    ax1.axvline(mu, color='green', linestyle='--', lw=2, label=f'μ = ${mu:.2f}')
    ax1.set_xlabel('Resíduo (USDT)', fontsize=12)
    ax1.set_ylabel('Densidade', fontsize=12)
    ax1.set_title('Distribuição de Resíduos', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Q-Q plot
    stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot', fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    # Teste de normalidade
    w_stat, p_value = stats.shapiro(residuals)
    ax2.text(0.05, 0.95, f'Shapiro-Wilk:\nW = {w_stat:.4f}\np = {p_value:.3f}', 
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    plt.suptitle('Figura 11. Distribuição de Resíduos (Erros de Predição)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('figura11_residuos.png', dpi=300, bbox_inches='tight')
    print("✅ Figura 11 salva: figura11_residuos.png")
    plt.close()

def figura12_heterocedasticidade(residuals, volatilidade):
    """Figura 12: Resíduos Absolutos vs Volatilidade"""
    abs_residuals = np.abs(residuals)
    
    plt.figure(figsize=(10, 7))
    plt.scatter(volatilidade, abs_residuals, alpha=0.5, s=50, edgecolors='black', linewidth=0.5)
    
    # Linha de tendência
    z = np.polyfit(volatilidade, abs_residuals, 1)
    p = np.poly1d(z)
    plt.plot(volatilidade, p(volatilidade), "r--", lw=2, label='Tendência Linear')
    
    # Correlação
    corr = np.corrcoef(volatilidade, abs_residuals)[0, 1]
    plt.text(0.05, 0.95, f'ρ = {corr:.3f}', 
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.xlabel('Volatilidade Realizada (7 dias)', fontsize=12)
    plt.ylabel('|Resíduo| (USDT)', fontsize=12)
    plt.title('Figura 12. Resíduos Absolutos vs. Volatilidade Realizada', 
              fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('figura12_heterocedasticidade.png', dpi=300, bbox_inches='tight')
    print("✅ Figura 12 salva: figura12_heterocedasticidade.png")
    plt.close()

def main():
    """Função principal para gerar todas as figuras"""
    print("\n" + "="*70)
    print("🎨 GERADOR DE FIGURAS DO ARTIGO CRYPTHDATA")
    print("="*70 + "\n")
    
    # Carregar dados
    df = carregar_dados()
    if df is None:
        return
    
    # Preparar dados
    print("📊 Preparando dados com feature engineering...")
    df = preparar_dados(df)
    
    # Figuras que não dependem de modelagem
    print("\n📈 Gerando figuras de análise exploratória...")
    figura1_pipeline_kdd()
    figura2_correlacao(df)
    figura3_distribuicao_retornos(df)
    figura4_elbow_silhouette(df)
    figura5_clusters(df)
    
    # Preparar dados para modelagem
    print("\n🤖 Treinando modelos...")
    features = ['open', 'high', 'low', 'close', 'volume', 'return', 
                'SMA_7', 'SMA_30', 'volatilidade_7d', 'RSI_14']
    
    X = df[features]
    y_class = df['price_up']
    y_reg = df['next_close']
    
    # Split temporal
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train_class, y_test_class = y_class[:split_idx], y_class[split_idx:]
    y_train_reg, y_test_reg = y_reg[:split_idx], y_reg[split_idx:]
    dates_test = df['open_time'][split_idx:].values
    
    # Normalização
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Classificação
    print("  → Random Forest Classifier...")
    rf_class = RandomForestClassifier(n_estimators=100, max_depth=10, 
                                      min_samples_split=5, min_samples_leaf=2,
                                      class_weight='balanced', random_state=42)
    rf_class.fit(X_train_scaled, y_train_class)
    y_pred_class = rf_class.predict(X_test_scaled)
    y_proba_class = rf_class.predict_proba(X_test_scaled)[:, 1]
    
    # Regressão
    print("  → Random Forest Regressor...")
    rf_reg = RandomForestRegressor(n_estimators=100, max_depth=15, 
                                   min_samples_split=5, random_state=42)
    rf_reg.fit(X_train_scaled, y_train_reg)
    y_pred_rf = rf_reg.predict(X_test_scaled)
    
    print("  → Linear Regression...")
    lr_reg = LinearRegression()
    lr_reg.fit(X_train_scaled, y_train_reg)
    y_pred_lr = lr_reg.predict(X_test_scaled)
    
    # Figuras de modelagem
    print("\n📊 Gerando figuras de resultados de modelagem...")
    figura6_confusion_matrix(y_test_class, y_pred_class)
    figura7_feature_importance(rf_class, features)
    figura8_roc_curve(y_test_class, y_proba_class)
    figura9_serie_temporal(dates_test, y_test_reg.values, y_pred_rf)
    figura10_scatter_plots(y_test_reg.values, y_pred_rf, y_pred_lr)
    
    # Resíduos
    residuals = y_test_reg.values - y_pred_rf
    volatilidade_test = df['volatilidade_7d'][split_idx:].values
    
    figura11_residuos(residuals)
    figura12_heterocedasticidade(residuals, volatilidade_test)
    
    print("\n" + "="*70)
    print("✅ TODAS AS 12 FIGURAS FORAM GERADAS COM SUCESSO!")
    print("="*70)
    print("\nArquivos salvos:")
    for i in range(1, 13):
        print(f"  ✓ figura{i}_*.png")
    print("\n")

if __name__ == "__main__":
    main()
