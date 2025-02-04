def select_X(X_y):
    """PLSRegression fit_transform iki çıktı döner, sadece X kısmını alır."""
    return X_y[0]  # İlk eleman olan X_transformed döndürülüyor

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
    
    future_predictions = []
    for period in periods:
        factor = simulated_factors.get(period, 1.0)
        predicted_price = base_pred * factor
        future_date = df['Date'].iloc[-1] + pd.Timedelta(days=period)
        future_predictions.append((future_date.strftime('%Y-%m-%d'), predicted_price))
    
    return future_predictions