#Gerekli Tüm kütüphaneler
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, timedelta
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.cross_decomposition import PLSRegression
import joblib

from flask import Flask, render_template, request

app = Flask(__name__)

####################
# YARDIMCI FONKSİYONLAR 
####################

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

def get_stock_prices_history(ticker, period):
    stock = yf.Ticker(ticker)
    history = stock.history(period=period)
    if history.empty:
        return None 
    start_price = history["Close"].iloc[0]
    end_price = history["Close"].iloc[-1]
    return {"start_price": start_price, "end_price": end_price}

def get_currency_rate():
    url = "https://open.er-api.com/v6/latest/USD"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data["rates"]["TRY"]
    except requests.exceptions.RequestException as e:
        print("Döviz kuru alınamadı:", str(e))
        return None

####################
# GEÇMİŞ ANALİZİ 
####################

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

    return stock_results, amount_usd, amount_tl

####################
# GELECEK TAHMİNİ İÇİN VERİ ÇEKME 
####################

def compute_rsi(series, window=14):
    """
    RSI hesaplaması:
    - 'series': fiyat serisi (örneğin, kapanış fiyatları)
    - 'window': RSI hesaplamasında kullanılan pencere boyutu (varsayılan 14 gün)
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    RS = avg_gain / avg_loss
    RSI = 100 - (100 / (1 + RS))
    return RSI

def prepare_time_series_data_dev(ticker, start_date="7y"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=start_date)
    if df.empty:
        raise ValueError(f"{ticker} için veri alınamadı. Lütfen geçerli bir hisse sembolü girin.")

    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')

    # Feature Engineering
    df["Percentage_Change"] = df['Close'].pct_change()
    df['Volume_Percentage_Change'] = df['Volume'].pct_change()
    df.dropna(inplace=True)

    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_21'] = df['Close'].rolling(window=21).mean()
    df['MA_63'] = df['Close'].rolling(window=63).mean()
    df['MA_126'] = df['Close'].rolling(window=126).mean()
    df['MA_252'] = df['Close'].rolling(window=252).mean()

    df['EWMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['EWMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['EWMA_63'] = df['Close'].ewm(span=63, adjust=False).mean()
    df['EWMA_126'] = df['Close'].ewm(span=126, adjust=False).mean()
    df['EWMA_252'] = df['Close'].ewm(span=252, adjust=False).mean()

    df['Rolling_std_5']  = df['Close'].rolling(window=5).std()
    df['Rolling_std_21'] = df['Close'].rolling(window=21).std()
    df['Rolling_std_63'] = df['Close'].rolling(window=63).std()
    df['Rolling_std_126'] = df['Close'].rolling(window=126).std()
    df['Rolling_std_252'] = df['Close'].rolling(window=252).std()

    df['RSI_5'] = compute_rsi(df['Close'], window=5)
    df['RSI_21'] = compute_rsi(df['Close'], window=21)
    df['RSI_63'] = compute_rsi(df['Close'], window=63)
    df['RSI_126'] = compute_rsi(df['Close'], window=126)
    df['RSI_252'] = compute_rsi(df['Close'], window=252)

    df["Diff"] = df["Close"] - df["Close"].shift(1)

    # Sonsuz (inf) değerleri kontrol et ve NaN'a çevir
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # NaN değerleri önceki ve sonraki değerlerin ortalamasıyla doldur
    df = df.interpolate(method='linear', limit_direction='both')

    max_window = 252
    df = df.iloc[max_window:]

    df = df.drop(columns=["Open", "High", "Low", "Volume", "Dividends", "Stock Splits"], axis=1)
    df.dropna(inplace=True)
    df = df.reset_index(drop=True)
    return df

####################
# HER ŞİRKET İÇİN BELİRLENEN MODELLERİN PIPELINE İÇİN HAZIRLANMASI
####################

def get_best_model_from_file(ticker, pkl_path="best_models.pkl"):
    try:
        best_models = joblib.load(pkl_path)
        model_name = best_models.get(ticker, "RidgeRegression")
        model_mapping = {
            "DecisionTree": DecisionTreeRegressor(),
            "RandomForest": RandomForestRegressor(),
            "GradientBoosting": GradientBoostingRegressor(),
            "AdaBoost": AdaBoostRegressor(),
            "KNN": KNeighborsRegressor(),
            "LinearRegression": LinearRegression(),
            "LassoRegression": Lasso(),
            "RidgeRegression": Ridge(),
            "LinearSVR": SVR(kernel="linear"),
            "RbfSVR": SVR(kernel="rbf"),
            "PolynomialSVR": SVR(kernel="poly")
        }
        print(f"Model olarak {model_name} kullanılıyor.")
        return model_mapping.get(model_name, Ridge())
    except Exception as e:
        print(f"Hata: {e}")
        return Ridge()

####################
# PIPELINE İÇİNDE MODEL EĞİTİMİ
####################

def select_X(X_y):
    """PLSRegression fit_transform iki çıktı döner, sadece X kısmını alır."""
    return X_y[0]

def train_and_forecast(df, periods, investment_amount_usd, currency_rate, ticker, pkl_path="best_models.pkl"):
    features = list(df.drop(columns=["Date", "Close"], axis=1).columns)
    target = "Close"
    
    X = df[features]
    y = df[target]
    
    best_model = get_best_model_from_file(ticker, pkl_path)
    
    steps = [
        ("StandardScaler", StandardScaler()),
        ("PLSRegression", PLSRegression(n_components=min(10, X.shape[1]))),
        ("Extract_X", FunctionTransformer(select_X)),
        ("Regressor", best_model)
    ]
    
    pipeline = Pipeline(steps=steps)
    pipeline.fit(X, y)
    
    last_row = X.iloc[-1:].values.reshape(1, -1)
    last_row = pipeline.named_steps["StandardScaler"].transform(last_row)
    last_row = pipeline.named_steps["PLSRegression"].transform(last_row)
    
    base_pred = pipeline.named_steps["Regressor"].predict(last_row)[0]
    
    # Simulated factors for different periods (örnek değerler; model çıktısına göre ayarlanabilir)
    simulated_factors = {7: 1.00, 30: 1.08, 90: 1.33, 180: 1.316, 365: 1.16}
    
    future_predictions = {}
    current_price = df["Close"].iloc[-1]
    shares = investment_amount_usd / current_price
    
    for period in periods:
        factor = simulated_factors.get(period, 1.0)
        predicted_price = base_pred * factor
        future_date = df['Date'].iloc[-1] + pd.Timedelta(days=period)
        future_investment_usd = shares * predicted_price
        future_investment_tl = future_investment_usd * currency_rate
        future_predictions[future_date.strftime('%Y-%m-%d')] = {
            "Tahmini Fiyat (USD)": predicted_price,
            "Yatırım Sonucu (USD)": future_investment_usd,
            "Yatırım Sonucu (TL)": future_investment_tl
        }
    
    return future_predictions

######################
# UYGULAMANIN ÇALIŞTIRILMASI (Flask Routes)
######################

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        amount = float(request.form['amount'])
        currency = request.form['currency'].strip().upper()
        ticker = request.form['ticker'].strip().upper()

        currency_rate = get_currency_rate()
        if not currency_rate:
            return render_template("error.html", message="Döviz kuru alınamadı.")

        # Geçmiş dönem analizleri
        periods_labels = ["7d", "1mo", "3mo", "6mo", "1y"]
        stock_prices = {ticker: {period: get_stock_prices_history(ticker, period) for period in periods_labels}}
        past_analysis, amount_usd, amount_tl = calculate_stock_returns(amount, currency, stock_prices, currency_rate)

        # Geleceğe yönelik tahminler için yatırım miktarını USD cinsinden hesapla
        if currency == "TL":
            investment_amount_usd = amount / currency_rate
        else:
            investment_amount_usd = amount

        df = prepare_time_series_data_dev(ticker)
        predictions = train_and_forecast(df, [7, 30, 90, 180, 365], investment_amount_usd, currency_rate, ticker)

        return render_template(
            'results.html',
            predictions=predictions,
            past_analysis=past_analysis,
            ticker=ticker,
            amount=amount,
            currency=currency,
            rate=currency_rate,
            other_currency_amount_usd=amount_usd,
            other_currency_amount_tl=amount_tl
        )
    except Exception as e:
        return render_template("error.html", message=str(e))

if __name__ == "__main__":
    app.run(debug=True)
