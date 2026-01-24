
import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

def main(args):
    X = np.load(args.input_path)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    np.save(args.output_features, X_scaled)
    joblib.dump(scaler, args.scaler_path)
    print("Preprocessing done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_features", required=True)
    parser.add_argument("--scaler_path", required=True)
    args = parser.parse_args()
    main(args)
