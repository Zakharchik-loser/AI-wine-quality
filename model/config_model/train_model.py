from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
import pandas as pd
import numpy as np
import joblib

df = pd.read_csv(r"C:\Users\zakha\PycharmProjects\Some_sort_of_ml\utils\WineQT.csv")



X = df.drop(["quality","Id"],axis=1)
y = df["quality"]


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1488)#I condemn this

model = XGBRegressor(
    booster="gbtree",
    objective="reg:squarederror",
    learning_rate = 0.1,
    reg_lambda = 1.0,
    reg_alpha = 0.0
)


model.fit(X_train,y_train)



y_pred = model.predict(X_test)


rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

joblib.dump(model,r"C:\Users\zakha\PycharmProjects\Some_sort_of_ml\model\config_model\HotChicken.pkl")


print(f"RMSE: {rmse:.3f}")
print(f"R²:   {r2:.3f}")
