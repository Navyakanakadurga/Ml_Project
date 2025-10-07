import argparse
import pandas as pd
def main(input_dir, output_file):
    # This script assumes cleaned CSVs already provided here; in production add robust cleaning steps
    files = [str(p) for p in Path(input_dir).glob("*.csv")]
    df_list = [pd.read_csv(f) for f in files]
    df = pd.concat(df_list, ignore_index=True)
    df.to_csv(output_file, index=False)
if __name__ == '__main__':
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    main(args.input, args.output)
