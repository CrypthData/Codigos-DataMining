"""
Script auxiliar para extrair dados do notebook eth.ipynb e salvar em CSV
Necessário para gerar as figuras do artigo
"""

import requests
import pandas as pd
import time

def get_full_binance_ohlc(symbol='ETHUSDT', interval='1d', start_date='2019-01-01'):
    """
    Coleta dados históricos completos da Binance com paginação automática
    """
    url = 'https://api.binance.com/api/v3/klines'
    
    # Converter data inicial para timestamp (milissegundos)
    start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
    end_ts = int(pd.Timestamp.now().timestamp() * 1000)
    
    all_data = []
    current_start = start_ts
    
    print(f"Coletando dados de {symbol} desde {start_date}...")
    
    while current_start < end_ts:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': current_start,
            'limit': 1000  # Máximo permitido pela API
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                break
            
            all_data.extend(data)
            
            # Atualizar timestamp para próxima requisição
            current_start = data[-1][0] + 1
            
            print(f"  Coletados {len(all_data)} registros até {pd.Timestamp(data[-1][0], unit='ms')}")
            
            # Delay para não sobrecarregar a API
            time.sleep(0.2)
            
        except Exception as e:
            print(f"Erro na requisição: {e}")
            break
    
    # Converter para DataFrame
    df = pd.DataFrame(all_data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'num_trades', 'taker_base_vol',
        'taker_quote_vol', 'ignore'
    ])
    
    # Converter tipos
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
    
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
    df[numeric_cols] = df[numeric_cols].astype(float)
    
    print(f"\n✅ Coleta finalizada: {len(df)} registros de {df['open_time'].min()} a {df['open_time'].max()}")
    
    return df

def main():
    print("="*70)
    print("📥 EXTRAÇÃO DE DADOS ETHEREUM (ETH/USDT)")
    print("="*70 + "\n")
    
    # Coletar dados
    df = get_full_binance_ohlc('ETHUSDT', '1d', '2019-01-01')
    
    # Salvar em CSV
    output_file = 'eth_data.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\n💾 Dados salvos em: {output_file}")
    print(f"📊 Total de observações: {len(df)}")
    print(f"📅 Período: {df['open_time'].min().date()} até {df['open_time'].max().date()}")
    
    print("\n✅ Agora você pode executar: python gerar_figuras_artigo.py")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
