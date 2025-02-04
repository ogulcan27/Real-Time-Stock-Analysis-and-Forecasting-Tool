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