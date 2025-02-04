def get_best_model_from_file(ticker, pkl_path="best_models.pkl"):
    try:
        best_models = joblib.load(pkl_path)
        model_name = best_models.get(ticker, "RidgeRegression")
        model_mapping = {"DecisionTree": DecisionTreeRegressor(),
                         "RandomForest": RandomForestRegressor(),
                         "GradientBoosting": GradientBoostingRegressor(),
                         "AdaBoost": AdaBoostRegressor(),
                         "KNN": KNeighborsRegressor(),
                         "LinearRegression": LinearRegression(),
                         "LassoRegression": Lasso(),
                         "RidgeRegression": Ridge(),
                         "LinearSVR": SVR(kernel="linear"),
                         "RbfSVR": SVR(kernel="rbf"),
                         "PolynomialSVR": SVR(kernel="poly")}
        print(f"Model olarak {model_name} kullanılıyor.")
        return model_mapping.get(model_name, Ridge())
    except Exception as e:
        print(f"Hata: {e}")
        return Ridge()