def main():
    amount = float(input("Yatırım yapmak istediğiniz miktar: "))
    currency = input("Para biriminiz nedir? (TL/USD): ").strip().upper()

    if currency not in ["TL", "USD"]:
        print("Geçersiz para birimi girdiniz. Lütfen 'TL' veya 'USD' olarak giriniz.")
        return

    currency_rate = get_currency_rate()
    if not currency_rate:
        print("Döviz kuru alınamadı, program sonlandırılıyor.")
        return

    if currency == "TL":
        investment_amount_usd = amount / currency_rate
        other_currency_amount = investment_amount_usd
        other_currency = "USD"
    else:
        investment_amount_usd = amount
        other_currency_amount = amount * currency_rate
        other_currency = "TL"

    ticker = input("Hangi hisse senedine yatırım yapmak istiyorsunuz? (örn: AAPL): ").strip().upper()

    try:
        df = prepare_time_series_data_dev(ticker)
        future_predictions = train_and_forecast(
            df,
            periods=[7, 30, 90, 180, 365],
            investment_amount_usd=investment_amount_usd,
            currency_rate=currency_rate,
            ticker=ticker
        )
    except ValueError as e:
        print(f"Hata: {e}")
        return

    periods_labels = ["7d", "1mo", "3mo", "6mo", "1y"]
    stock_prices = {ticker: {period: get_stock_prices_history(ticker, period) for period in periods_labels}}
    stock_results, amount_usd, amount_tl = calculate_stock_returns(amount, currency, stock_prices, currency_rate)

    print("\nYatırım Analiz ve Tahmin Raporu\n")
    print("Girdiğiniz Bilgiler:")
    print(f"  Yatırım Miktarı: {amount:.2f} {currency}")
    print(f"  {other_currency} Karşılığı: {other_currency_amount:.2f} {other_currency} (1 USD = {currency_rate:.2f} TL)")
    print("\nGeçmişten Günümüze Yatırım Analizi:")

    for ticker_key, periods_data in stock_results.items():
        print(f"\nHisse: {ticker_key}")
        for period_label, result in periods_data.items():
            date_range = get_period_date_range(period_label)
            print(f"  Dönem: {period_label} - {date_range}")
            if isinstance(result, str):
                print(f"    {result}")
            else:
                for key, value in result.items():
                    if isinstance(value, float):
                        print(f"    {key}: {value:.2f}")
                    else:
                        print(f"    {key}: {value}")

    print("\nGeleceğe Yönelik Tahminler:")
    # Mevcut kapanış fiyatını alarak yatırım miktarının kaç hisse ettiğini hesaplayalım.
    current_price = df["Close"].iloc[-1]
    shares = investment_amount_usd / current_price

    for future_date, predicted_price in future_predictions:
        investment_result_usd = shares * predicted_price
        investment_result_tl = investment_result_usd * currency_rate
        print(f"  Tarih: {future_date}")
        print(f"    Tahmini Fiyat (USD): {predicted_price:.2f}")
        print(f"    Yatırım Sonucu (USD): {investment_result_usd:.2f}")
        print(f"    Yatırım Sonucu (TL): {investment_result_tl:.2f}")

if __name__ == "__main__":
    main()