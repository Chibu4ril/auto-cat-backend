import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import uuid



if len(sys.argv) < 2:
    print("Error: No file URL provided")
    sys.exit(1)

file_url = sys.argv[1] 




# file_path = "https://fyfpcpjfjuqroiwjdgjk.supabase.co/storage/v1/object/public/uploads/D19-com.csv?"  # Replace with the actual file path

# Read CSV file into DataFrame
df = pd.read_csv(file_url)

# Make sure that the date column is in datetime format
df['Created Date'] = pd.to_datetime(df['Created Date'], dayfirst=True)  # dayfirst=True to avoid warning


# Convert 'Attended' to numeric values (if not already)
df['Attended'] = pd.to_numeric(df['Attended'], errors='coerce')  # Convert non-numeric values to NaN

# Handle missing or NaN values in 'Attended' by replacing them with 0 (not attended)
df['Attended'] = df['Attended'].fillna(0)




# Extract week number from 'Created Date' for trend analysis
df['Week Number'] = df['Created Date'].dt.isocalendar().week

# Assuming each booking is a new registration, calculate cumulative registrations
df['Cumulative Registrations'] = df.groupby('Week Number')['BookingReference'].transform('count').cumsum()

# Calculate the number of attendees (handle NaN values correctly)
df['Cumulative Attendees'] = df.groupby('Week Number')['Attended'].transform('sum').cumsum()

# Now, we prepare x (week number) and y (cumulative registrations) for logistic growth model
x = df['Week Number'].unique()  # Unique week numbers for x-axis
y = df.groupby('Week Number')['BookingReference'].count().cumsum().values  # Cumulative registrations

# Logistic growth function
def logisticGrowth(t, K, N0, r):
    denom = 1 + ((K - N0) / N0) * np.exp(-r * t)
    return K / denom

# Initial guess for the parameters [K, N0, r]
p0 = [max(y) * 1.2, y[0], 0.1]

# Fit the logistic growth model
popt, _ = curve_fit(logisticGrowth, x, y, p0=p0, maxfev=10000)

# Extract the optimal parameters
K, N0, r = popt

# Generate future weeks for prediction
futureWeeks = np.arange(len(x), len(x) + 10)
futurePredictions = logisticGrowth(futureWeeks, K, N0, r)

# Plot the results
output_file = f'Logistic_Growth_Model_{str(uuid.uuid4())}.png'
plt.figure(figsize=(10, 5))
plt.scatter(x, y, label='Actual Data', color='blue')
plt.plot(x, logisticGrowth(x, K, N0, r), label='Fitted Logistic Curve', color='green')
plt.plot(futureWeeks, futurePredictions, label='Future Predictions', linestyle='dashed', color='red')
plt.xlabel('Week')
plt.ylabel('Cumulative Registrations')
plt.title(f'Logistic Growth Model - {str(uuid.uuid4())}')
plt.legend()
plt.savefig(output_file, format='png', dpi=300)

# Return result as JSON (output)
output_data = {
    "prediction_plot": output_file,
    "future_predictions": futurePredictions.tolist()
}

# Print output (for testing)
print(json.dumps(output_data, indent=4))
