import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# WEATHER PATTERN ANALYSIS
# ==========================================

# Load dataset
file_path = "india_weather_rainfall_small.csv"
df = pd.read_csv(file_path)

# ==========================================
# 1. BASIC DATA INFORMATION
# ==========================================

print("========== WEATHER PATTERN ANALYSIS ==========")

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

# ==========================================
# 2. DATA PREPROCESSING
# ==========================================

# Convert date column
df["date_of_record"] = pd.to_datetime(
    df["date_of_record"],
    errors="coerce"
)

# Convert numeric columns
numeric_columns = [
    "rainfall",
    "avg_temp",
    "min_temp",
    "max_temp",
    "wind_speed",
    "air_pressure"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

# Remove rows without date or rainfall
df = df.dropna(
    subset=["date_of_record", "rainfall"]
)

# Create year and month
df["year"] = df["date_of_record"].dt.year
df["month_num"] = df["date_of_record"].dt.month

# ==========================================
# 3. RAINFALL STATISTICS
# ==========================================

print("\n========== RAINFALL STATISTICS ==========")

print(
    "Total Recorded Rainfall:",
    df["rainfall"].sum()
)

print(
    "Average Rainfall:",
    df["rainfall"].mean()
)

print(
    "Maximum Rainfall:",
    df["rainfall"].max()
)

print(
    "Minimum Rainfall:",
    df["rainfall"].min()
)

# ==========================================
# 4. YEARLY RAINFALL ANALYSIS
# ==========================================

yearly_rainfall = (
    df.groupby("year")["rainfall"]
    .sum()
)

print("\n========== YEARLY RAINFALL ==========")
print(yearly_rainfall)

plt.figure(figsize=(10, 5))

yearly_rainfall.plot(
    kind="bar"
)

plt.title("Yearly Rainfall")
plt.xlabel("Year")
plt.ylabel("Total Rainfall")
plt.tight_layout()
plt.show()

# ==========================================
# 5. MONTHLY AVERAGE RAINFALL
# ==========================================

monthly_rainfall = (
    df.groupby("month_num")["rainfall"]
    .mean()
    .reindex(range(1, 13))
)

month_names = [
    "Jan", "Feb", "Mar", "Apr",
    "May", "Jun", "Jul", "Aug",
    "Sep", "Oct", "Nov", "Dec"
]

monthly_rainfall.index = month_names

print("\n========== MONTHLY AVERAGE RAINFALL ==========")
print(monthly_rainfall)

plt.figure(figsize=(10, 5))

monthly_rainfall.plot(
    kind="bar"
)

plt.title("Monthly Average Rainfall")
plt.xlabel("Month")
plt.ylabel("Average Rainfall")
plt.tight_layout()
plt.show()

# ==========================================
# 6. STATE-WISE RAINFALL ANALYSIS
# ==========================================

state_rainfall = (
    df.groupby("state")["rainfall"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

print("\n========== TOP 10 STATES BY AVERAGE RAINFALL ==========")
print(state_rainfall)

plt.figure(figsize=(10, 5))

state_rainfall.plot(
    kind="bar"
)

plt.title("Top 10 States by Average Rainfall")
plt.xlabel("State")
plt.ylabel("Average Rainfall")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# ==========================================
# 7. TEMPERATURE TREND
# ==========================================

yearly_temperature = (
    df.groupby("year")["avg_temp"]
    .mean()
)

print("\n========== AVERAGE TEMPERATURE BY YEAR ==========")
print(yearly_temperature)

plt.figure(figsize=(10, 5))

yearly_temperature.plot(
    kind="line",
    marker="o"
)

plt.title("Average Temperature Trend")
plt.xlabel("Year")
plt.ylabel("Average Temperature")

plt.tight_layout()
plt.show()

# ==========================================
# 8. MINIMUM AND MAXIMUM TEMPERATURE
# ==========================================

temp_by_year = (
    df.groupby("year")[
        ["min_temp", "max_temp"]
    ].mean()
)

plt.figure(figsize=(10, 5))

plt.plot(
    temp_by_year.index,
    temp_by_year["min_temp"],
    marker="o",
    label="Minimum Temperature"
)

plt.plot(
    temp_by_year.index,
    temp_by_year["max_temp"],
    marker="o",
    label="Maximum Temperature"
)

plt.title(
    "Minimum and Maximum Temperature Trend"
)

plt.xlabel("Year")
plt.ylabel("Temperature")

plt.legend()
plt.tight_layout()
plt.show()

# ==========================================
# 9. RAINFALL VS AVERAGE TEMPERATURE
# ==========================================

plt.figure(figsize=(8, 5))

plt.scatter(
    df["avg_temp"],
    df["rainfall"],
    alpha=0.5
)

plt.title(
    "Rainfall vs Average Temperature"
)

plt.xlabel("Average Temperature")
plt.ylabel("Rainfall")

plt.tight_layout()
plt.show()

# ==========================================
# 10. SEASON-WISE RAINFALL
# ==========================================

season_rainfall = (
    df.groupby("season")["rainfall"]
    .mean()
)

print("\n========== SEASON-WISE AVERAGE RAINFALL ==========")
print(season_rainfall)

plt.figure(figsize=(8, 5))

season_rainfall.plot(
    kind="bar"
)

plt.title(
    "Season-wise Average Rainfall"
)

plt.xlabel("Season")
plt.ylabel("Average Rainfall")

plt.tight_layout()
plt.show()

# ==========================================
# PROJECT COMPLETED
# ==========================================

print(
    "\nWeather Pattern Analysis Completed Successfully!"
)
```
