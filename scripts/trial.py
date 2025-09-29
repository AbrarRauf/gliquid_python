import pandas as pd

output_path = "all_dumps/gliq_manu_test3/output.csv"

df = pd.read_csv(output_path)
print(df.head())

# calculate average RMSE between columns "melting_temperature_in_kelvin" and "gliq_melting_temp_kelvin"
rmse = ((df["melting_temperature_in_kelvin"] - df["mpds_melting_point_kelvin"]) ** 2).mean() ** 0.5
print(rmse)

# compute the standard deviation of the differences between the two columns
std_dev = (df["melting_temperature_in_kelvin"] - df["mpds_melting_point_kelvin"]).std()
print(std_dev)

# make a scatter plot of the two columns with mpds in x-axis and the other in y-axis
# show a dashed line y=x
import matplotlib.pyplot as plt
plt.scatter(df["mpds_melting_point_kelvin"], df["melting_temperature_in_kelvin"], color='tab:cyan')
plt.plot([0, 3000], [0, 3000], 'k--', label="y=x",)
plt.xlabel("MPDS Melting Point (K)")
plt.ylabel("MAPP Prediction Melting Point (K)")
plt.xlim(250, 2500)
plt.ylim(250, 2500)
plt.legend()
plt.title("MPDS vs Fitted and Interpolated Melting Points")
plt.show()