"""
Script para gerar a Tabela 1 do artigo como imagem
Tabela: Etapas de Pré-Processamento Aplicadas
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def criar_tabela1():
    """Cria a Tabela 1 - Etapas de Pré-Processamento"""
    
    # Dados da tabela
    dados = [
        ['Seleção de Atributos', 
         'Remoção de quote_volume, num_trades,\ntaker_base_vol, taker_quote_vol, ignore', 
         'Redundância informacional e\nbaixa relevância preditiva'],
        
        ['Conversão de Tipos', 
         'open_time → datetime64[ns];\npreços → float64', 
         'Padronização para processamento\ntemporal e numérico'],
        
        ['Ordenação Temporal', 
         'Ordenação ascendente por open_time', 
         'Garantir sequência cronológica\npara séries temporais'],
        
        ['Detecção de Valores\nAusentes', 
         'Verificação de NaN via isna().sum()', 
         'Identificar lacunas na\nsérie temporal'],
        
        ['Tratamento de Outliers', 
         'Inspeção visual via boxplots,\nsem remoção automática', 
         'Outliers em cripto refletem\neventos reais (crashes, pumps)'],
        
        ['Validação de\nConsistência', 
         'Verificação: low ≤ open/close ≤ high\npara todas as observações', 
         'Garantir integridade lógica\ndos dados OHLC']
    ]
    
    # Configurar figura
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # Criar tabela
    tabela = ax.table(cellText=dados,
                      colLabels=['Etapa', 'Procedimento', 'Justificativa'],
                      cellLoc='left',
                      loc='center',
                      colWidths=[0.20, 0.45, 0.35])
    
    # Estilizar tabela
    tabela.auto_set_font_size(False)
    tabela.set_fontsize(11)
    tabela.scale(1, 3)
    
    # Estilizar cabeçalho
    for i in range(3):
        cell = tabela[(0, i)]
        cell.set_facecolor('#2C3E50')
        cell.set_text_props(weight='bold', color='white', fontsize=13)
        cell.set_height(0.08)
    
    # Estilizar células
    cores_alternadas = ['#ECF0F1', '#FFFFFF']
    for i in range(1, len(dados) + 1):
        for j in range(3):
            cell = tabela[(i, j)]
            cell.set_facecolor(cores_alternadas[i % 2])
            cell.set_edgecolor('#BDC3C7')
            cell.set_linewidth(1.5)
            
            # Negrito na primeira coluna
            if j == 0:
                cell.set_text_props(weight='bold', fontsize=11)
            
            # Código monospace na segunda coluna
            if j == 1:
                cell.set_text_props(fontfamily='monospace', fontsize=10)
    
    # Título
    plt.suptitle('Tabela 1. Etapas de Pré-Processamento Aplicadas', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Nota de rodapé
    nota = "Após pré-processamento, o dataset manteve 2.535 registros íntegros sem valores ausentes no período analisado."
    plt.figtext(0.5, 0.02, nota, ha='center', fontsize=11, 
                style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig('tabela1_preprocessamento.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print("✅ Tabela 1 salva: tabela1_preprocessamento.png")
    plt.close()

if __name__ == "__main__":
    print("\n" + "="*70)
    print("📊 GERANDO TABELA 1 - ETAPAS DE PRÉ-PROCESSAMENTO")
    print("="*70 + "\n")
    
    criar_tabela1()
    
    print("\n" + "="*70)
    print("✅ TABELA GERADA COM SUCESSO!")
    print("="*70 + "\n")
