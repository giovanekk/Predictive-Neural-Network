import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans



# Get data and clean it
df23 = pd.read_csv('data/23-24.csv', delimiter=';')
#drop last row of df23:
df23 = df23.drop(df23.index[-1])
df24 = pd.read_csv('data/24-25.csv', delimiter=';')
df = pd.concat([df23, df24])
df = df.drop(columns=['Pedido','Hora','Representante','Status','Observa��o','Cliente Desde',
                      'Devol','Cidade','UF','numero','numero2','telefone','NF','Method'])
df = df.dropna(axis=1, how='all') #drop all columns that are all NaN
df = df.dropna(axis=0, how='all') #this might drop everything
#translate column names to english
df = df.rename(columns={'Data':'Date','Vendedor':'Seller',
                        'Quantidade':'Quantity',
                        'Total Bruto':'Gross', 'Pagamento':'Payment Method',
                        'Desconto':'Discount','Frete':'Shipping',
                        'datadecompra':'Purchase Date','Devolu��o':'Returned','L�quido':'Net',
                        'Cliente':'Client','Qtde':'Quantity'})

# Make all commas into dots (replace commas with dots)
df = df.replace(',', '.', regex=True) #replace commas with dots

# Make non numeric columns numeric (Data, Vendedor, Produto, Pagamento, UF, Cidade, Cliente)

#Date (make numeric)
# Convert to datetime
df['Purchase Date'] = pd.to_datetime(df['Purchase Date'], format='%d/%m/%Y')
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

# Get the most recent date in dataset
latest_date = df['Purchase Date'].max()

# Compute client "age" in days
client_age = df.groupby('Client')['Date'].min().apply(lambda d: (latest_date - d).days)

# Merge client age back into full dataframe
df['Client Age'] = df['Client'].map(client_age)

df = df.drop(columns=['Date']) #drop date columns

# Purchase Date (turn into Last Purchase in days)
df['Purchase Date'] = pd.to_datetime(df['Purchase Date'], format='%d/%m/%Y')
df= df.sort_values(by=['Client','Purchase Date'])
df['Last Purchase'] = df.groupby('Client')['Purchase Date'].diff().dt.days
df = df.drop(columns=['Purchase Date']) #drop purchase date column

#Seller (remove 04 and then one hot encode)
df = df[df.Seller != '04']
print(df["Seller"].unique()) #check unique sellers
df = pd.get_dummies(df, columns=['Seller'], prefix='Seller')

#Payment Method (one hot encoding; for now don't include)
df = df.drop(columns=['Payment Method']) #drop payment method column

#Clients
# There are more than 600 clients, so we will not one hot encode them. 
# Instead, we will plot the Gross, Quantity, and Date features of each client and 
# use K means clustering to categorize them.
# Using these new categories, we will one hot encode them.

# Find any cell that literally contains the problematic string
# Convert critical columns to numeric safely
for col in ['Gross', 'Quantity', 'Client Age']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df[df.Client != ''] #drop empty clients
# Show rows that were invalid
bad_rows = df[df.isna().any(axis=1)]
if not bad_rows.empty:
    print("Dropping rows with malformed numbers:")
    print(bad_rows[['Client', 'Gross', 'Quantity', 'Client Age']])

# Drop rows that still contain NaNs after coercion
df = df.dropna(subset=['Gross', 'Quantity', 'Client Age'])

client_features = df.groupby('Client').agg({
    'Gross': 'sum',
    'Quantity': 'sum',
    'Client Age': 'first'  
}).fillna(0)

X_client = StandardScaler().fit_transform(client_features)
kmeans = KMeans(n_clusters=6, random_state=45)
clusters = kmeans.fit_predict(X_client)

# Map clusters back to original df
client_cluster_map = dict(zip(client_features.index, clusters))
df['Client Cluster'] = df['Client'].map(client_cluster_map)
df = df.drop(columns=['Client']) #drop client column
df = pd.get_dummies(df, columns=['Client Cluster'], prefix='Cluster')

df = df.dropna()
print(df.head())
# Get input data
X = df.drop(columns=['Gross']) #drop gross column
X = X.astype(np.float32) #convert to float32
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# Get target data
y = df['Gross'] #get gross column
y = y.astype(np.float32) #convert to float32
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()


# turn into tensors
y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

X_tensor = torch.tensor(X.values, dtype=torch.float32)

dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

print(torch.isnan(X_tensor).any(), torch.isinf(X_tensor).any(), X_tensor.max(), X_tensor.min())
print(torch.isnan(y_tensor).any(), torch.isinf(y_tensor).any(), y_tensor.max(), y_tensor.min())



# Define the neural network model (TRIANGLE ARCHITECTURE)
# This is a simple feedforward neural network with one hidden layer
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(X_tensor.shape[1], 300)  # Input layer
        self.fc2 = nn.Linear(300, 150) # Hidden layer
        self.fc3 = nn.Linear(150, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

model = NeuralNetwork()

#second model (STABLENET)
# This is a more complex feedforward neural network with batch normalization and dropout
class StableNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)
    
model2 = StableNet(X_tensor.shape[1])



# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model2.parameters(), lr=0.0001)

# Prepare data for training
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Train the model
losses = []
epochs = 50
for epoch in range(epochs):
    epoch_losses = []
    for batch_X, batch_y in dataloader:
        # Forward pass
        predictions = model2(batch_X)
        loss = criterion(predictions, batch_y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())

    # Store average loss for the whole epoch
    avg_loss = sum(epoch_losses) / len(epoch_losses)
    losses.append(avg_loss)

    # Print loss for every epoch
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
    
# Save the model
torch.save(model.state_dict(), 'prediction_model.pth')

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(losses) + 1), losses, marker='o')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss")
plt.grid(True)
plt.tight_layout()
plt.show()

