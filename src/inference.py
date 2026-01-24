
import argparse
import numpy as np
import tensorflow as tf

def main(args):
    encoder = tf.keras.models.load_model(args.encoder_path, compile=False)
    lstm = tf.keras.models.load_model(args.lstm_path, compile=False)

    X = np.load(args.features_path)
    latent = encoder.predict(X)

    X_seq = np.array([latent[i:i+args.window] for i in range(len(latent)-args.window)])
    preds = lstm.predict(X_seq).flatten()

    diff = np.diff(preds)
    print("Mean abs temporal gradient:", np.mean(np.abs(diff)))
    print("Monotonicity violation:", np.mean(diff > 0))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_path", required=True)
    parser.add_argument("--encoder_path", required=True)
    parser.add_argument("--lstm_path", required=True)
    parser.add_argument("--window", type=int, default=20)
    args = parser.parse_args()
    main(args)
