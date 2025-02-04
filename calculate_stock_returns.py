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
                "Başlangıç Fiyatı (USD)": data["start_price"],
                "Bitiş Fiyatı (USD)": data["end_price"],
                "Değişim Oranı (%)": change_rate * 100,
                "Sonuç (USD)": result_usd,
                "Sonuç (TL)": result_tl
            }

    return stock_results, amount_usd, amount_tl