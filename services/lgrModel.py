import os
import sys
import json
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import logging
from sklearn.metrics import mean_absolute_percentage_error
import uuid

logging.basicConfig(filename="/tmp/script_output.log", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# print(sys.argv)
if len(sys.argv) < 2:
    print("Error: Missing required arguments. Usage: script.py <file_url>")
    sys.exit(1)

# testdate = "18/10/2024"


file_url = sys.argv[1] 

# Read CSV and process dates
df = pd.read_csv(file_url, skiprows=1)

# print(df.head())



df['Created Date'] = pd.to_datetime(df['Created Date'], dayfirst=True, errors='coerce')
# print(df['Created Date'])


# Drop rows with invalid dates
df = df.dropna(subset=['Created Date'])

# Sort by created date
df = df.sort_values(by='Created Date')

# Calculate start date and week number
min_created_date = df['Created Date'].min()
max_created_date = df['Created Date'].max()



df['Week Number'] = ((df['Created Date'] - min_created_date).dt.days // 7) + 1

# Calculate weeks until the event
weeks_to_event = ((max_created_date + pd.Timedelta(days=21)) - min_created_date).days // 7


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
popt, _ = curve_fit(logisticGrowth, x, y, p0 = p0, maxfev = 10000)



K, N0, r = popt

adjusted_r = r  # Store original value
if len(y) >= 3 and np.all(y[-3:] == y[-1]):  
    adjusted_r *= 1.2  # Increase growth rate slightly
    r = adjusted_r  # Apply adjustment


# Split data: 80% training and 20% test
split_idx = int(len(x) * 0.8)
x_train, x_test = x[:split_idx], x[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]


# Train model on 80% of the data
popt_train, _ = curve_fit(logisticGrowth, x_train, y_train, p0=p0, maxfev=10000)


# Predict for the test set
y_pred = logisticGrowth(x_test, *popt_train)


# Calculate Mean Absolute Percentage Error (MAPE)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100  # Convert to %
prediction_accuracy = round(100 - mape, 2)


futureWeeks = np.arange(len(x), weeks_to_event + 1)
futurePredictions = logisticGrowth(futureWeeks, K, N0, r)


adjusted_growth_rate = r 


if len(y) >= 3 and np.all(y[-3:] == y[-1]):   # If last 3 weeks had no increase
    adjusted_growth_rate = r * 1.2  # Slightly increase growth rate
    growth_message = (
        "The model detected a plateau in the last 3 weeks of registrations. "
        "The growth rate (r) has been slightly increased to account for potential "
        "last-minute sign-ups. This could indicate a sudden surge in registrations as the event date approaches."
    )
elif r < 0.1:  # If growth rate is very low
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
    "final_predicted_registrations": int(futurePredictions[-1]),  # Last week's prediction
    "prediction_accuracy_percent": prediction_accuracy,  # Accuracy score
    "weeks_until_event": weeks_to_event,
    "initial_growth_rate_r": round(popt[2], 4),
    "adjusted_growth_rate_r": round(adjusted_growth_rate, 4),
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





# plt.figure(figsize = (10, 5))
# plt.scatter(x, y, label = 'Actual Data', color = 'blue')
# plt.plot(x, logisticGrowth(x, K, N0, r), label = 'Fitted Logistic Curve', color = 'green')
# plt.plot(futureWeeks, futurePredictions, label = 'Future Predictions', linestyle = 'dashed', color = 'red')
# plt.scatter(futureWeeks, futurePredictions, color='red', marker='x', label='Future Points', zorder=6)
# plt.xlabel('Week')
# plt.ylabel('Cumulative Registrations')
# plt.title('Logistic Growth Model for Cumulative Registrations')
# plt.legend()
# plt.grid(True)


# output_file = f'Logistic_Growth_Model_{str(uuid.uuid4())}.png'

# plt.savefig(output_file, format = 'png', dpi = 300)

# Return result as JSON (output)


output_data = {
    # "prediction_plot": output_file
    "x": x.tolist(),  # x-values (e.g., weeks or time points)
    "y": y.tolist(),  # Actual data
    "futureWeeks": futureWeeks.tolist(),  # Future weeks
    "futurePredictions": futurePredictions.tolist(),  # Future predictions
    # "logistic_growth_values": logistic_growth_values.tolist(),  # Logistic growth values based on the logistic model
    "parameters": {
        "K": K,   # Carrying capacity
        "N0": N0, # Initial population
        "r": r    # Growth rate
    },
    "weeks": [f"Week {i + 1}" for i in range(len(futurePredictions))],
    "future_predictions": futurePredictions.tolist(),
    "metadata": metadata
}


def sanitize_data(obj):
    """ Recursively replace NaN, None, and infinite values in the JSON output. """
    if isinstance(obj, dict):
        return {k: sanitize_data(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_data(i) for i in obj]
    elif isinstance(obj, (int, float)):
        if np.isnan(obj) or np.isinf(obj):
            return 0  # Replace NaN/inf with 0
        return obj
    return obj

output_data = sanitize_data(output_data)

logging.debug(f"Output Data: {output_data}")

try:
    output_json = json.dumps(output_data)
    print(output_json)  # For debugging
except Exception as e:
    print(f"JSON Serialization Error: {e}")

