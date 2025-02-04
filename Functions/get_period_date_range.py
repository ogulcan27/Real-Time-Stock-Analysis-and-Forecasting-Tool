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