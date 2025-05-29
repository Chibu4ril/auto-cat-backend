import sys
import json
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import logging

logging.basicConfig(filename="/tmp/script_output.log", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

if len(sys.argv) < 2:
    print("Error: Missing required argument. Usage: script.py <file_url>")
    sys.exit(1)

file_url = sys.argv[1] 

df = pd.read_csv(file_url, skiprows=1)

df['Created Date'] = pd.to_datetime(df['Created Date'], dayfirst=True, errors='coerce')

# Drop rows with invalid dates
df = df.dropna(subset=['Created Date'])

# Sort by created date
df = df.sort_values(by='Created Date')

# Calculate start date and week number
min_created_date = df['Created Date'].min()
df['Week Number'] = ((df['Created Date'] - min_created_date).dt.days // 7) + 1

# Extend the date range by adding 2 weeks to the maximum created date
max_created_date = df['Created Date'].max() + pd.Timedelta(weeks=2)

# Identify registration statuses
df['Is Registration'] = df['Attendee Status'].isin(['Attending', 'Cancelled'])

# Calculate cumulative registrations per week
weeklyCounts = df.groupby('Week Number')['Is Registration'].sum().cumsum().reset_index()
weeklyCounts.rename(columns={'Is Registration': 'Cumulative Registrations'}, inplace=True)

# Ensure all week numbers are included
allWeeks = pd.DataFrame({'Week Number': range(1, weeklyCounts['Week Number'].max() + 1)})
weeklyCounts = allWeeks.merge(weeklyCounts, on='Week Number', how='left')

# Fill missing cumulative registrations with the previous week's value
weeklyCounts.loc[:, 'Cumulative Registrations'] = weeklyCounts['Cumulative Registrations'].ffill()

weeklyCounts['Week'] = np.arange(len(weeklyCounts))

x = weeklyCounts['Week Number'].values
y = weeklyCounts['Cumulative Registrations'].values

if len(y) == 0 or np.all(np.isnan(y)):
    logging.error("Error: Cumulative Registrations is empty or contains only NaN values.")
    sys.exit(1)

def logisticGrowth(t, K, N0, r):
    denom = 1 + ((K - N0) / N0) * np.exp(-r * t)
    return K / denom

p0 = [max(y) * 1.2, y[0], 0.1]

# Split data: 80% training and 20% test
split_idx = int(len(x) * 0.8)
x_train, x_test = x[:split_idx], x[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Train model on 80% of the data
popt_train, _ = curve_fit(logisticGrowth, x_train, y_train, p0=p0, maxfev=10000)
# popt_train, _ = curve_fit(logisticGrowth, x, y, p0=p0, maxfev=10000)

# Predict for the test set
y_pred = logisticGrowth(x, *popt_train)

# Calculate Mean Absolute Percentage Error (MAPE)
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    nonzero_indices = y_true != 0  # Avoid division by zero
    return np.mean(np.abs((y_true[nonzero_indices] - y_pred[nonzero_indices]) / y_true[nonzero_indices])) * 100

mape = mean_absolute_percentage_error(y, y_pred)  # Convert to %
prediction_accuracy = round(100 - mape, 2)

K, N0, r = popt_train
futureWeeks = np.arange(len(x), len(x) + 5)  # Predicting 5 future weeks
futurePredictions = logisticGrowth(futureWeeks, K, N0, r)

# Dynamic response for growth rate and prediction accuracy
adjusted_growth_rate = r
if len(y) >= 3 and np.all(y[-3:] == y[-1]):
    adjusted_growth_rate = r * 1.2
    growth_message = (
        "The model detected a plateau in the last 3 weeks of registrations. "
        "The growth rate (r) has been slightly increased to account for potential "
        "last-minute sign-ups. This could indicate a sudden surge in registrations as the event date approaches."
    )
elif r < 0.1:
    growth_message = (
        "The model shows a very slow registration growth. "
        "This could indicate that there is low interest or insufficient marketing efforts so far. "
        "You may need to consider ramping up marketing activities to boost registration numbers."
    )
else:
    growth_message = (
        "The model predicts steady and consistent growth in registrations over time. "
        "This is a positive sign, indicating that interest in the event is growing steadily as the event date approaches."
    )



if prediction_accuracy < 85:
    accuracy_message = (
        f"The prediction model has a low accuracy of {prediction_accuracy}%. "
        "This suggests that the data may not fully capture the expected growth, "
        "or that there may be other factors affecting registration patterns."
    )
elif prediction_accuracy < 95:
    accuracy_message = (
        f"The prediction model has a moderate accuracy of {prediction_accuracy}%. "
        "While this provides a reasonable estimate, further validation may be needed to refine the predictions."
    )
else:
    accuracy_message = (
        f"The prediction model is highly accurate with an accuracy of {prediction_accuracy}%. "
        "This indicates that the model has closely followed the actual trends in registration."
    )

metadata = {
    "final_predicted_registrations": int(futurePredictions[-1]),
    "prediction_accuracy_percent": prediction_accuracy,
    "weeks_until_event": len(futureWeeks),
    "initial_growth_rate_r": round(r, 4),
    "adjusted_growth_rate_r": round(adjusted_growth_rate, 4),
    "growth_message": accuracy_message,
    "graph_interpretation": (
        f"The logistic growth model predicts cumulative registrations leading up to the event. "
        f"Initially, registration growth follows a slow pace, but as time progresses, "
        f"registrations are expected to increase at a faster rate due to growing interest.\n\n"
        f"{growth_message}\n\n"
        f"{accuracy_message}\n\n"
        f"The model suggests that, depending on the trends, the event might see either steady or sudden "
        f"surges in registrations, particularly in the final weeks leading to the event. "
        f"The predicted final registration number is {int(futurePredictions[-1])}."
    )
}

logistic_growth_values = logisticGrowth(x, K, N0, r)

output_data = {
    "x": x.tolist(),
    "y": y.tolist(),
    "futureWeeks": futureWeeks.tolist(),
    "futurePredictions": futurePredictions.tolist(),
    "parameters": {"K": K, "N0": N0, "r": r},
     "weeks": [f"Week {i + 1}" for i in range(len(futurePredictions))],
    "metadata": metadata,
    "logistic_growth_values": logistic_growth_values.tolist(),  # Logistic growth values based on the logistic model
    "mape": mape

}

def sanitize_data(obj):
    if isinstance(obj, dict):
        return {k: sanitize_data(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_data(i) for i in obj]
    elif isinstance(obj, (int, float)):
        if np.isnan(obj) or np.isinf(obj):
            return 0
        return obj
    elif isinstance(obj, str):
        return obj
    return obj

output_data = sanitize_data(output_data)

logging.debug(f"Output Data: {output_data}")

try:
    output_json = json.dumps(output_data, indent=4)
    print(output_json)
except Exception as e:
    print(f"JSON Serialization Error: {e}")
