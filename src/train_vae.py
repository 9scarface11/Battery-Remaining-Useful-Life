
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

def build_vae(input_dim, latent_dim):
    inputs = layers.Input(shape=(input_dim,))
    h = layers.Dense(32, activation="relu")(inputs)
    z_mean = layers.Dense(latent_dim)(h)
    z_log_var = layers.Dense(latent_dim)(h)

    def sampling(args):
        z_mean, z_log_var = args
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * eps

    z = layers.Lambda(sampling)([z_mean, z_log_var])
    encoder = Model(inputs, z_mean)

    latent_inputs = layers.Input(shape=(latent_dim,))
    h_dec = layers.Dense(32, activation="relu")(latent_inputs)
    outputs = layers.Dense(input_dim)(h_dec)
    decoder = Model(latent_inputs, outputs)

    outputs = decoder(z)
    vae = Model(inputs, outputs)
    vae.compile(optimizer="adam", loss="mse")

    return vae, encoder, decoder

def main(args):
    X = np.load(args.features_path)
    vae, encoder, decoder = build_vae(X.shape[1], args.latent_dim)
    vae.fit(X, X, epochs=args.epochs, batch_size=args.batch_size)
    encoder.save(args.encoder_path)
    decoder.save(args.decoder_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_path", required=True)
    parser.add_argument("--encoder_path", required=True)
    parser.add_argument("--decoder_path", required=True)
    parser.add_argument("--latent_dim", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    main(args)
