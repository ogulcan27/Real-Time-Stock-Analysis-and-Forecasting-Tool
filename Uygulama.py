from flask import Flask, render_template, request
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import requests
import pandas as pd
import numpy as np

app = Flask(__name__)

# Döviz kuru alma
def get_currency_rate():
    url = "https://open.er-api.com/v6/latest/USD"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data["rates"]["TRY"]
    except requests.exceptions.RequestException:
        return None

# Hisse geçmişi alma
def get_stock_prices_history(ticker, period):
    stock = yf.Ticker(ticker)
    history = stock.history(period=period)
    if history.empty:
        return None
    start_price = history['Close'].iloc[0]
    end_price = history['Close'].iloc[-1]
    return {"start_price": start_price, "end_price": end_price}

# Tarih aralığı hesaplama
def get_period_date_range(period):
    end_date = datetime.today()
    if period == "7d":
        start_date = end_date - timedelta(days=7)
    elif period == "1mo":
        start_date = end_date - timedelta(days=30)
    elif period == "3mo":
        start_date = end_date - timedelta(days=90)
    elif period == "6mo":
        start_date = end_date - timedelta(days=180)
    elif period == "1y":
        start_date = end_date - timedelta(days=365)
    else:
        start_date = None

    return f"{start_date.strftime('%d.%m.%Y')} - {end_date.strftime('%d.%m.%Y')}" if start_date else "Tarih bilgisi hesaplanamadı"

# Geçmiş analiz hesaplama
def calculate_stock_returns(amount, currency, stock_prices, currency_rate):
    if currency == "TL":
        amount_usd = amount / currency_rate
    else:
        amount_usd = amount

    amount_tl = amount_usd * currency_rate

    stock_results = {}
    for ticker, periods in stock_prices.items():
        stock_results[ticker] = {}
        for period, data in periods.items():
            if not data:
                stock_results[ticker][period] = "Veri bulunamadı"
                continue

            change_rate = (data["end_price"] - data["start_price"]) / data["start_price"]

            result_usd = amount_usd * (1 + change_rate)
            result_tl = result_usd * currency_rate

            stock_results[ticker][period] = {
                "Tarih Aralığı": get_period_date_range(period),
                "Başlangıç Fiyatı (USD)": data["start_price"],
                "Bitiş Fiyatı (USD)": data["end_price"],
                "Değişim Oranı (%)": change_rate * 100,
                "Sonuç (USD)": result_usd,
                "Sonuç (TL)": result_tl
            }

    return stock_results

#Geleceğe yönelik tahminler için veri setini hazırlama ve feature eng
def prepare_time_series_data(ticker, start_date="7y"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=start_date)
    if df.empty:
        raise ValueError(f"{ticker} için veri alınamadı. Lütfen geçerli bir hisse sembolü girin.")

    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')

    # Feature Engineering
    df["Percentage_Change"] = ((df['Close'] - df['Open']) / df['Open']) * 100
    df['Volume_Percentage_Change'] = df['Volume'].pct_change() * 100
    df.dropna(inplace=True)

    df['MA_7'] = df['Close'].rolling(window=7).mean()
    df['MA_30'] = df['Close'].rolling(window=30).mean()
    df['MA_60'] = df['Close'].rolling(window=60).mean()
    df['MA_90'] = df['Close'].rolling(window=90).mean()
    df['MA_180'] = df['Close'].rolling(window=180).mean()
    df["MA_365"] = df["Close"].rolling(window=365).mean()

    df['EWMA_7'] = df['Close'].ewm(span=7, adjust=False).mean()
    df['EWMA_30'] = df['Close'].ewm(span=30, adjust=False).mean()
    df['EWMA_60'] = df['Close'].ewm(span=60, adjust=False).mean()
    df['EWMA_90'] = df['Close'].ewm(span=90, adjust=False).mean()
    df['EWMA_180'] = df['Close'].ewm(span=180, adjust=False).mean()
    df['EWMA_365'] = df['Close'].ewm(span=365, adjust=False).mean()

    df = df.drop(columns=["Open", "High", "Low", "Volume", "Dividends", "Stock Splits"], axis=1)
    df.dropna(inplace=True)
    df = df.reset_index(drop=True)
    return df

#Pipeline ile Tahmin
def train_and_forecast(df, periods, investment_amount_usd, currency_rate):
    df.dropna(inplace=True)

    features = list(df.drop(columns=["Date", "Close"], axis=1).columns)
    target = 'Close'

    X = df[features]
    y = df[target]

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', Ridge())
    ])

    pipeline.fit(X, y)

    last_row = df.iloc[-1][features].values.reshape(1, -1)
    future_predictions = {}
    for period in periods:
        future_date = df['Date'].iloc[-1] + pd.Timedelta(days=period)
        predicted_price = pipeline.predict(last_row)[0]
        future_investment_usd = investment_amount_usd * (predicted_price / df['Close'].iloc[-1])
        future_investment_tl = future_investment_usd * currency_rate

        future_predictions[future_date.strftime('%Y-%m-%d')] = {
            "Tahmini Fiyat (USD)": predicted_price,
            "Yatırım Sonucu (USD)": future_investment_usd,
            "Yatırım Sonucu (TL)": future_investment_tl
        }
        last_row = np.roll(last_row, -1)
        last_row[0, -1] = predicted_price

    return future_predictions

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        amount = float(request.form['amount'])
        currency = request.form['currency']
        ticker = request.form['ticker']

        currency_rate = get_currency_rate()
        if not currency_rate:
            return render_template("error.html", message="Döviz kuru alınamadı.")

        # Geçmiş dönem analizleri
        periods = ["7d", "1mo", "3mo", "6mo", "1y"]
        stock_prices = {ticker: {period: get_stock_prices_history(ticker, period) for period in periods}}
        past_analysis = calculate_stock_returns(amount, currency, stock_prices, currency_rate)

        # Geleceğe yönelik tahminler
        investment_amount_usd = amount / currency_rate if currency == "TL" else amount
        investment_amount_tl = amount if currency == "TL" else amount * currency_rate
        df = prepare_time_series_data(ticker)
        predictions = train_and_forecast(
            df, [7, 30, 90, 180, 365], investment_amount_usd, currency_rate
        )

        return render_template(
            'results.html',
            predictions=predictions,
            past_analysis=past_analysis,
            ticker=ticker,
            amount=amount,
            currency=currency,
            rate=currency_rate,
            other_currency_amount_usd=investment_amount_usd,
            other_currency_amount_tl=investment_amount_tl
        )
    except Exception as e:
        return render_template("error.html", message=str(e))

if __name__ == "__main__":
    app.run(debug=True)
