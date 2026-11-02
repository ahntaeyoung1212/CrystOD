import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(
        description="Derive the mean value of T, E, V in md.csv for a given range"
    )
    parser.add_argument(
        "--start", type=int, required=True,
        help="Start row number (1-based index)"
    )
    parser.add_argument(
        "--end", type=int, required=True,
        help="End row number (1-based index, inclusive)"
    )
    parser.add_argument(
        "--file", type=str, default="md.csv",
        help="Input CSV file name (default: md.csv)"
    )

    args = parser.parse_args()

    # Load the CSV file
    df = pd.read_csv(args.file)

    # Convert from 1-based indexing to 0-based indexing
    start_idx = args.start - 1
    end_idx = args.end  # iloc end index is exclusive

    # Extract mean volume 
    Etot = df.iloc[start_idx:end_idx, 4]
    T = df.iloc[start_idx:end_idx, 5]
    V = df.iloc[start_idx:end_idx, 9]
    a = df.iloc[start_idx:end_idx, 6]
    b = df.iloc[start_idx:end_idx, 7]
    c = df.iloc[start_idx:end_idx, 8]

    # Compute the mean value
    mean_Etot = Etot.mean()
    mean_T = T.mean()
    mean_V = V.mean()
    mean_a = a.mean()
    mean_b = b.mean()
    mean_c = c.mean()

    print(f"Mean value of Etot (eV): {mean_Etot}")
    print(f"Mean value of T (K):     {mean_T}")
    print(f"Mean value of V (A^3):   {mean_V}")
    print(f"Mean value of a (A):     {mean_a}")
    print(f"Mean value of b (A):     {mean_b}")
    print(f"Mean value of c (A):     {mean_c}")

if __name__ == "__main__":
    main()


# import pandas as pd
# 
# df = pd.read_csv("md.csv")
# 
# V_values = df.iloc[2000:30000, 9]
# 
# mean_V = V_values.mean()
# 
# print(f"{mean_V} A^3")
