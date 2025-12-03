import pandas as pd

df = pd.read_csv(r"C:\Users\zakha\PycharmProjects\Some_sort_of_ml\utils\WineDataset.csv")

def safe(x):
    return "" if pd.isna(x) else str(x)

df["text"] = df.apply(
    lambda row: (
        f"Title: {safe(row['Title'])}\n"
        f"Description: {safe(row['Description'])}\n"
        f"Grape Varieties: {safe(row['Grape'])}, {safe(row['Secondary Grape Varieties'])}\n"
        f"Country: {safe(row['Country'])}, Region: {safe(row['Region'])}, Appellation: {safe(row['Appellation'])}\n"
        f"Style: {safe(row['Style'])}\n"
        f"Characteristics: {safe(row['Characteristics'])}\n"
        f"ABV: {safe(row['ABV'])}\n"
        f"Price: {safe(row['Price'])}\n"
        f"Capacity: {safe(row['Capacity'])}"
    ),
    axis=1
)


if __name__ == "__main__":
    print(df[["Description","text"]].head())
