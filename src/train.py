from data_loader import load_data
from preprocessing import preprocess
from model import build_vae_lstm

def main():
    data = load_data('data/sample_data.csv')
    X = preprocess(data)
    X = X.reshape((X.shape[0], 1, X.shape[1]))
    model = build_vae_lstm(X.shape[1:])
    model.fit(X, data['RUL'], epochs=5)

if __name__ == '__main__':
    main()
