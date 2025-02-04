def get_stock_prices_history(ticker, period):
    stock = yf.Ticker(ticker)
    history = stock.history(period=period)
    if history.empty:
        return None 
    start_price = history["Close"].iloc[0]
    end_price = history["Close"].iloc[-1]
    return {"start_price": start_price, "end_price": end_price}