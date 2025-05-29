
# import requests
# import pandas as pd
# from xgboost import XGBClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split
# from xgboost import XGBRegressor
# from sklearn.metrics import mean_absolute_error
# import matplotlib.pyplot as plt
# import seaborn as sns



# databucket = "http://localhost:8000/api/getFormData"

# # Send GET request to FastAPI to fetch the FormData
# response = requests.get(databucket, verify=False)

# if response.status_code == 200:
#     # Get the FormData response from FastAPI
#     form_data = response.json()
#     print("Form Data received from FastAPI:")
#     print(form_data)
    
    
#     event_name = form_data.get("eventName")

#     # Load dataset from server
#     file_path = form_data.get("selectedFileUrl")
    
#     if file_path:
#         print(f"📂 Dataset file URL: {file_path}")

#         # Load dataset from the file URL
#         df = pd.read_csv(file_path)

#         # Convert 'Created Date' to datetime format (Assuming it's in day/month/year format)
#         df['Created Date'] = pd.to_datetime(df['Created Date'], dayfirst=True, errors='coerce')

#         # # Drop unnecessary columns (update if needed)
#         df.drop(columns=['Reference'], errors='ignore', inplace=True)

#         # # Remove duplicates based on Booking Reference (keeping latest entry)
#         df = df.sort_values(by='Created Date').drop_duplicates(subset='BookingReference', keep='last')

#          # Convert categorical columns to numerical values
#         df['Attendee Status'] = df['Attendee Status'].map({'Attending': 1, 'Cancelled': 0, 'Booker not attending': 0})
#         df['Attended'] = df['Attended'].map({'Yes': 1, 'No': 0}).fillna(0)

#         print("📊 Dataset Loaded Successfully!")
#         print(df.head())  
#     else:
#         print("No file URL found in FormData!")

# else:
#     print(f"Error fetching FormData: {response.status_code}")
    


    
   
    
   
    
   
    
#     # # Feature: Calculate 'Days From Start'
#     # df['Days From Start'] = (df['Created Date'] - df['Created Date'].min()).dt.days
    
#     # # External factors (Placeholder values, update with real data)
#     # df['Marketing Budget'] = 5000  
#     # df['Public Holiday Nearby'] = 0  
#     # df['COVID Impact'] = 0  
    
#     # # Assign 'Event ID' if missing
#     # if 'Event ID' not in df.columns:
#     #     df['Event ID'] = df['Event Name'].astype('category').cat.codes if 'Event Name' in df.columns else 1
        
#     #     print(df.head())  # Check cleaned data

# # event_df = df.groupby('Event ID').agg({
# #     'Attended': 'sum',
# #     'BookingReference': 'count',
# #     'Marketing Budget': 'first',
# #     'Public Holiday Nearby': 'first',
# #     'COVID Impact': 'first',
# #     'Days From Start': 'max'
# # }).rename(columns={'BookingReference': 'Total Registrations'}).reset_index()

# # print(event_df.head())  # Verify event-level dataset



# # # Define features and target
# # features_reg = ['Marketing Budget', 'Public Holiday Nearby', 'COVID Impact', 'Days From Start']
# # X_reg = event_df[features_reg]
# # y_reg = event_df['Total Registrations']

# # # Handle small datasets: If only 1 event, train on full data
# # if len(event_df) > 1:
# #     X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.1, random_state=42)
# # else:
# #     X_train_reg, X_test_reg, y_train_reg, y_test_reg = X_reg, X_reg, y_reg, y_reg

# # # Train XGBoost Regressor
# # reg_model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
# # reg_model.fit(X_train_reg, y_train_reg)

# # # Evaluate model
# # y_pred_reg = reg_model.predict(X_test_reg)
# # mae = mean_absolute_error(y_test_reg, y_pred_reg)
# # print(f"Registration Prediction MAE: {mae:.2f}")




# # # Define features and target
# # features_cls = ['Attendee Status', 'Days From Start', 'Marketing Budget', 'Public Holiday Nearby', 'COVID Impact']
# # X_cls = df[features_cls]
# # y_cls = df['Attended']

# # # Handle small datasets
# # if len(df) > 1:
# #     X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, test_size=0.1, random_state=42)
# # else:
# #     X_train_cls, X_test_cls, y_train_cls, y_test_cls = X_cls, X_cls, y_cls, y_cls

# # # Train XGBoost Classifier
# # cls_model = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, use_label_encoder=False, eval_metric="logloss", random_state=42)
# # cls_model.fit(X_train_cls, y_train_cls)

# # # Evaluate model
# # y_pred_cls = cls_model.predict(X_test_cls)
# # accuracy = accuracy_score(y_test_cls, y_pred_cls)
# # print(f"Attendance Prediction Accuracy: {accuracy * 100:.2f}%")



# # # Registration Model Feature Importance
# # plt.figure(figsize=(8,6))
# # sns.barplot(x=reg_model.feature_importances_, y=features_reg)
# # plt.xlabel("Importance")
# # plt.ylabel("Feature")
# # plt.title("Feature Importance for Registration Prediction")
# # plt.show()

# # # Attendance Model Feature Importance
# # plt.figure(figsize=(8,6))
# # sns.barplot(x=cls_model.feature_importances_, y=features_cls)
# # plt.xlabel("Importance")
# # plt.ylabel("Feature")
# # plt.title("Feature Importance for Attendance Prediction")
# # plt.show()