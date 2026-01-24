
import argparse
import numpy as np

def main(args):
    data = np.load(args.input_path)
    np.save(args.output_path, data)
    print("Data loaded and saved")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()
    main(args)
