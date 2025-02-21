import pandas as pd
from datetime import datetime
import streamlit as st
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import yfinance as yf
# Configuración de Streamlit
st.set_page_config(layout="wide")

# Función para extracción de datos históricos
def extract_historical_data(ticker, start_date='2022-01-01'):
    df = yf.download(ticker, start=start_date, progress=False)
    df.reset_index(inplace=True)
    df['Crypto'] = ticker.split('-')[0]
    df = df[['Date', 'Close', 'Crypto']]
    df.rename(columns={'Date': 'Timestamp', 'Close': 'Actual Price'}, inplace=True)
    return df

# Transformación de datos
def transform_data(df):
    df['Highest 1H'] = df['Actual Price'].rolling(window=6).max()
    df['Lower 1H'] = df['Actual Price'].rolling(window=6).min()
    df['AVG Price'] = df['Actual Price'].rolling(window=6).mean()
    df['24hr_Change'] = df['Actual Price'].pct_change(periods=1).fillna(0) * 100
    df['Signal'] = df['24hr_Change'].apply(lambda x: 'B' if x > 0 else 'S')
    return df

# Pronóstico con Regresión Polinómica
def forecast_polynomial(df):
    df['Time_Index'] = np.arange(len(df))
    X = df[['Time_Index']]
    y = df['Actual Price']

    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)
    forecast_index = np.array([[len(df) + i] for i in range(1, 31)])
    forecast_poly = poly.transform(forecast_index)
    forecast_prices = model.predict(forecast_poly)

    return pd.Series(forecast_prices.ravel()), model

# Pronóstico con ARIMA
def forecast_arima(df):
    model = ARIMA(df['Actual Price'], order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=30)
    return pd.Series(forecast.values.ravel()), model_fit

# Pronóstico con SARIMA
def forecast_sarima(df):
    model = SARIMAX(df['Actual Price'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit(disp=False)
    forecast = model_fit.forecast(steps=30)
    return pd.Series(forecast.values.ravel()), model_fit

# Visualización de los datos históricos
def plot_historical_data(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df['Timestamp'], df['Actual Price'], label='Actual Price', color='blue')
    plt.plot(df['Timestamp'], df['Highest 1H'], label='Highest 1H', linestyle='--', color='green')
    plt.plot(df['Timestamp'], df['Lower 1H'], label='Lower 1H', linestyle='--', color='red')
    plt.plot(df['Timestamp'], df['AVG Price'], label='AVG Price', linestyle=':', color='orange')
    plt.title('Historical Price Data')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

# Unir gráficos de pronósticos
def plot_combined_forecast(df, poly_forecast, arima_forecast, sarima_forecast):
    future_dates = [df['Timestamp'].iloc[-1] + pd.Timedelta(days=i) for i in range(1, 31)]
    plt.figure(figsize=(14, 7))
    plt.plot(df['Timestamp'], df['Actual Price'], label='Actual Price', color='blue')
    plt.plot(future_dates, poly_forecast, label='Polynomial Regression', linestyle='--', color='purple')
    plt.plot(future_dates, arima_forecast, label='ARIMA', linestyle='--', color='green')
    plt.plot(future_dates, sarima_forecast, label='SARIMA', linestyle='--', color='red')
    plt.title('Forecast Comparison')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

# Recomendación de compra o venta
def generate_recommendation(selected_date, forecasts):
    avg_forecast = np.mean([forecasts['Polynomial'][selected_date],
                            forecasts['ARIMA'][selected_date],
                            forecasts['SARIMA'][selected_date]])
    if avg_forecast > forecasts['Polynomial'][selected_date - 1]:
        return "Compra: Se espera una tendencia al alza."
    else:
        return "Venta: Se espera una tendencia a la baja."

# Streamlit App
def streamlit_app():
    st.title("PHAROS - Plataforma de Previsión de Operaciones con Yahoo Finanzas")

    cryptos = [
    'BTC-USD',   # Bitcoin
    'ETH-USD',   # Ethereum
    'XRP-USD',   # Ripple
    'LTC-USD',   # Litecoin
    'DOGE-USD',  # Dogecoin
    'ADA-USD',   # Cardano
    'SOL-USD',   # Solana
    'DOT-USD',   # Polkadot
    'AVAX-USD',  # Avalanche
    'MATIC-USD', # Polygon
    'BNB-USD',   # Binance Coin
    'USDT-USD',  # Tether
    'USDC-USD',  # USD Coin
    'BCH-USD',   # Bitcoin Cash
    'LINK-USD',  # Chainlink
    'UNI-USD',   # Uniswap
    'XLM-USD',   # Stellar
    'ATOM-USD',  # Cosmos
    'VET-USD',   # VeChain
    'ALGO-USD',  # Algorand
    'FTM-USD',   # Fantom
    'ICP-USD',   # Internet Computer
    'FIL-USD',   # Filecoin
    'AAVE-USD',  # Aave
    'SAND-USD',  # The Sandbox
    'MANA-USD',  # Decentraland
    'GRT-USD',   # The Graph
    'XTZ-USD',   # Tezos
    'EOS-USD',   # EOS
    'THETA-USD', # Theta Network
    'LUNA1-USD', # Terra 2.0
    'TRX-USD',   # TRON
    'NEAR-USD',  # NEAR Protocol
    'HNT-USD',   # Helium
    'CHZ-USD',   # Chiliz
    'ENJ-USD',   # Enjin Coin
    'ZIL-USD',   # Zilliqa
    'RUNE-USD',  # THORChain
    'KSM-USD',   # Kusama
    'STX-USD',   # Stacks
    'GALA-USD',  # Gala
    'DYDX-USD',  # dYdX
    'CRV-USD',   # Curve DAO Token
    '1INCH-USD', # 1inch Network
    'SNX-USD',   # Synthetix
    'BAND-USD',  # Band Protocol
    'OCEAN-USD', # Ocean Protocol
    'KAVA-USD',  # Kava
    'ANKR-USD',  # Ankr
    'TWT-USD',   # Trust Wallet Token
    'COMP-USD',  # Compound
    'CAKE-USD',  # PancakeSwap
    'RAY-USD',   # Raydium
    'MOVR-USD',  # Moonriver
    'XEM-USD',   # NEM
    'BTG-USD',   # Bitcoin Gold
    'DASH-USD',  # Dash
    'RVN-USD',   # Ravencoin
    'KNC-USD',   # Kyber Network
    'LRC-USD',   # Loopring
    'AR-USD',    # Arweave
    'CFX-USD',   # Conflux
    'GNO-USD',   # Gnosis
    'PERP-USD',  # Perpetual Protocol
    'YFI-USD',   # yearn.finance
    'COTI-USD',  # COTI
    'UMA-USD',   # UMA
    'RSR-USD',   # Reserve Rights
    'CELR-USD',  # Celer Network
    'CTSI-USD',  # Cartesi
    'XDC-USD',   # XDC Network
    'ELF-USD',   # aelf
    'MTL-USD',   # Metal
    'OM-USD',    # MANTRA DAO
    'JST-USD',   # JUST
    'REEF-USD',  # Reef
    'IDEX-USD',  # IDEX
    'RAD-USD',   # Radicle
    'DNT-USD'    # district0x
]
    selected_crypto = st.selectbox("Selecciona la Criptomoneda", cryptos, index=0)

    data = extract_historical_data(selected_crypto, start_date='2022-01-01')
    transformed_data = transform_data(data)

    st.write("### Datos Históricos")
    plot_historical_data(transformed_data)

    st.write("### Datos Resumidos")
    st.dataframe(transformed_data[['Timestamp', 'Actual Price', 'Highest 1H', 'Lower 1H', 'AVG Price']].tail(10),
                 use_container_width=True)

    # Pronósticos
    poly_forecast, poly_model = forecast_polynomial(transformed_data)
    arima_forecast, arima_model = forecast_arima(transformed_data)
    sarima_forecast, sarima_model = forecast_sarima(transformed_data)

    # Gráficos combinados
    st.write("### Comparación de Pronósticos")
    plot_combined_forecast(transformed_data, poly_forecast, arima_forecast, sarima_forecast)

    # Recomendación personalizada
    st.write("### Recomendación Personalizada")
    selected_date = st.slider("Selecciona un día dentro del rango pronosticado", min_value=1, max_value=30)
    forecasts = {
        'Polynomial': poly_forecast,
        'ARIMA': arima_forecast,
        'SARIMA': sarima_forecast
    }
    recommendation = generate_recommendation(selected_date, forecasts)
    st.write(f"Recomendación para el día {selected_date}: **{recommendation}**")

if __name__ == '__main__':
    streamlit_app()
