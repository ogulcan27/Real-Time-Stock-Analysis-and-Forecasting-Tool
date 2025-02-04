def compute_rsi(series, window=14):
    """
    RSI hesaplaması:
    - 'series': fiyat serisi (örneğin, kapanış fiyatları)
    - 'window': RSI hesaplamasında kullanılan pencere boyutu (varsayılan 14 gün)
    """
    delta = series.diff()
    # Yalnızca kazanç ve kayıpları ayrı ayrı hesaplayalım
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Basit hareketli ortalama ile hesaplama (alternatif olarak EWMA da kullanılabilir)
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

    #Sonsuz (inf) değerleri kontrol et ve NaN'a çevir
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    #NaN değerleri önceki ve sonraki değerlerin ortalamasıyla doldur
    df = df.interpolate(method='linear', limit_direction='both')

    #!!!!
    max_window = 252
    df = df.iloc[max_window:]

    df = df.drop(columns=["Open", "High", "Low", "Volume", "Dividends", "Stock Splits"], axis=1)
    df.dropna(inplace=True)
    df = df.reset_index(drop=True)
    return df