import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# Carregar o dataset
file_path = 'complete_dataset.csv'
dataset = pd.read_csv(file_path)

# Limpeza dos dados
dataset['solar_exposure'].fillna(dataset['solar_exposure'].mean(), inplace=True)
dataset['rainfall'].fillna(dataset['rainfall'].mean(), inplace=True)

# Converter colunas categóricas
dataset['school_day'] = dataset['school_day'].apply(lambda x: 1 if x == 'Y' else 0)
dataset['holiday'] = dataset['holiday'].apply(lambda x: 1 if x == 'Y' else 0)

scaler_demand = MinMaxScaler()
dataset['demand'] = scaler_demand.fit_transform(dataset[['demand']])

# Normalizar os dados
scaler = MinMaxScaler()
columns_to_normalize = dataset.columns[1:]  # Excluir 'date'
dataset[columns_to_normalize] = scaler.fit_transform(dataset[columns_to_normalize])

# Separar as features e o target
X = dataset.drop(columns=['date', 'demand'])
y = dataset['demand']

# Dividir em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Construir a rede neural
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, input_dim=X_train.shape[1], activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
])

# Compilar o modelo
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Treinar o modelo
n_epocs = 1000
n_batch = 16
n_validation = 0.1
n_verbose = 2

history = model.fit(X_train, y_train, epochs=n_epocs, batch_size=n_batch, validation_split=n_validation, verbose=n_verbose)

# Avaliar o modelo
loss, mae = model.evaluate(X_test, y_test, verbose=1)


# Fazer previsões
predictions = model.predict(X_test)

# Criar um scaler apenas para a coluna 'demand' para inverter a normalização
# Criar um scaler apenas para a coluna 'demand' para inverter a normalização
#scaler_demand = MinMaxScaler()
#scaler_demand.fit(dataset[['demand']])
#scaler_demand = scaler

# Denormalizar as previsões e os valores reais usando o scaler da coluna 'demand'
predictions_denormalized = scaler_demand.inverse_transform(predictions)
y_test_denormalized = scaler_demand.inverse_transform(y_test.values.reshape(-1, 1))

# Calcular o erro absoluto entre as previsões e os valores reais
errors = abs(predictions_denormalized - y_test_denormalized)

# Imprimir as previsões, os valores reais e os erros
for pred, actual, error in zip(predictions_denormalized, y_test_denormalized, errors):
    print(f"VALOR OBTIDO: {pred[0]} - VALOR ESPERADO: {actual[0]} - ERRO: {error[0]}")
print(f"Mean Absolute Error: {mae}")
print(f"Numero de épocas: {n_epocs} | Batch: {n_batch} | Validation: {n_validation} | Verbose: {n_verbose}")