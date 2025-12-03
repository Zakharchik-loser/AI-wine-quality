import pandas as pd

df = pd.read_csv(r"C:\Users\zakha\PycharmProjects\Some_sort_of_ml\utils\WineQT.csv")



df["text"] = df.apply(
    lambda row: (
        f"Wine {row['Id']}: Fixed Acidity {row['fixed acidity']}, "
        f"Volatile Acidity {row['volatile acidity']}, "
        f"Sugar {row['residual sugar']}, pH {row['pH']}, "
        f"Sulfates {row['sulphates']}, Alcohol {row['alcohol']}, "
        f"Quality {row['quality']}."
    ),
    axis=1
)




if __name__ == "__main__":
        print(df[["Id","text"]].head())