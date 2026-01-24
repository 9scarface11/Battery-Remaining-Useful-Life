
import argparse
import numpy as np
import tensorflow as tf

def main(args):
    X = np.load(args.latent_path)
    y = np.load(args.rul_path)

    X_seq, y_seq = [], []
    for i in range(len(X) - args.window):
        X_seq.append(X[i:i+args.window])
        y_seq.append(y[i+args.window])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(32, input_shape=X_seq.shape[1:]),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(X_seq, y_seq, epochs=args.epochs, batch_size=args.batch_size)
    model.save(args.model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_path", required=True)
    parser.add_argument("--rul_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    main(args)
