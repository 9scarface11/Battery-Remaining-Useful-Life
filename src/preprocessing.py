from sklearn.preprocessing import MinMaxScaler

def preprocess(df):
    scaler = MinMaxScaler()
    X = df.drop(columns=['RUL'])
    return scaler.fit_transform(X)
