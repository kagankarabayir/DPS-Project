import urllib.request
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pickle
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

url = 'https://opendata.muenchen.de/api/3/action/datastore_search?resource_id=40094bd6-f82d-4979-949b-26c8dc00b9a7&limit=5000'
response = urllib.request.urlopen(url)
data = json.loads(response.read().decode())

records = data['result']['records']
df = pd.DataFrame(records)

df = df[['MONATSZAHL', 'AUSPRAEGUNG', 'JAHR', 'MONAT', 'WERT']]
filtered_data = df[(df['MONATSZAHL'] == 'Alkoholunfälle') & (df['AUSPRAEGUNG'] == 'insgesamt')].copy()
filtered_data['JAHR'] = pd.to_numeric(filtered_data['JAHR'])
filtered_data = filtered_data[filtered_data['JAHR'] <= 2020]
filtered_data = filtered_data[filtered_data['MONAT'] != 'Summe']
filtered_data['MONAT'] = filtered_data['MONAT'].astype(str).str[-2:].astype(int)
filtered_data['JAHR'] = pd.to_numeric(filtered_data['JAHR'], errors='coerce')
filtered_data['MONAT'] = pd.to_numeric(filtered_data['MONAT'], errors='coerce')
filtered_data = filtered_data.dropna(subset=['JAHR', 'MONAT'])
filtered_data = filtered_data[(filtered_data['MONAT'] >= 1) & (filtered_data['MONAT'] <= 12)]
filtered_data['DATUM'] = filtered_data.apply(lambda row: pd.Timestamp(year=int(row['JAHR']), month=int(row['MONAT']), day=1), axis=1)

fig = px.line(filtered_data, x='DATUM', y='WERT', title='Alcohol-related Accidents: Historical Accident Data', labels={'DATUM': 'Date', 'WERT': 'Number of Accidents'}, markers=True)
fig.update_traces(line=dict(width=2))
fig.update_layout(template="plotly_white", title_x=0.5)
fig.show()

filtered_data['Month_Index'] = filtered_data['JAHR'] * 12 + filtered_data['MONAT']
X = filtered_data[['Month_Index']]
y = filtered_data['WERT']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

y_train_log = np.log1p(y_train)
model = LinearRegression()
model.fit(X_train, y_train_log)

with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

# Ensuring the predicted value is not negative
jan_2021_index = (2021 * 12) + 1
jan_2021_index_scaled = scaler.transform([[jan_2021_index]])
predicted_value_log = model.predict(jan_2021_index_scaled)
predicted_value = np.expm1(predicted_value_log)
print(f"Predicted number of accidents for 'Alcohol-related Accidents', 'Overall' in January 2021: {max(predicted_value[0], 0)}")  # Added max function to handle negative values

y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_pred = np.maximum(y_pred, 0)  
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error of the model: {mse}")

fig = go.Figure()
fig.add_trace(go.Scatter(x=X_test.flatten(), y=y_test, mode='markers', name='Observations', marker=dict(color='blue')))
fig.add_trace(go.Scatter(x=X_test.flatten(), y=y_pred, mode='markers', name='Predictions', marker=dict(color='red')))
fig.update_layout(title='Prediction vs Observation', xaxis_title='Month Index', yaxis_title='Number of Accidents', legend=dict(x=0.01, y=0.99), template="plotly_white", title_x=0.5)
fig.show()