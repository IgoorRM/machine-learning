import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.layers import Dense
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import classification_report

df1 = pd.read_excel (r'C:\Users\Iguzz\Documents\Codes\PYTHON\base.xlsx')

# Pré-processamento de dados
label_encoder = LabelEncoder()
df1['formacao'] = label_encoder.fit_transform(df1['formacao'])

print(df1)

# Identificar as colunas de matérias
materias_cols = [col for col in df1.columns if col not in ['professor', 'formacao']]

# Aplicar a codificação one-hot apenas no DataFrame df1_encoded
onehot_encoder = OneHotEncoder(sparse=False)
y_encoded = onehot_encoder.fit_transform(df1[['formacao']])

# Adicionar colunas de rótulos one-hot ao DataFrame original
df1_encoded = pd.concat([df1, pd.DataFrame(y_encoded, columns=[f'formacao_{int(i)}' for i in range(y_encoded.shape[1])])], axis=1)

# Extrair características (X) e rótulos (y) do DataFrame df1_encoded
X = df1_encoded.drop(['professor'] + materias_cols, axis=1)
y = y_encoded

# Divisão em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Criação do modelo
model = Sequential()
model.add(Dense(12, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(y_train.shape[1], activation='softmax'))  # Número de neurônios igual ao número de classes

# Compilação do modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinamento do modelo
history = model.fit(X_train, y_train, epochs=50, batch_size=1, verbose=1, validation_data=(X_test, y_test))

# Avaliação do modelo no conjunto de teste
accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f'Acurácia do Modelo: {accuracy[1]*100:.2f}%')

# Função para imprimir os resultados
def print_results(professors, predicted_subjects, true_subjects):
    results_df = pd.DataFrame({
        'professor': professors,
        'Matéria Prevista': predicted_subjects,
        'Matéria Real': true_subjects
    })
    print(results_df)
    return results_df

# Previsões no conjunto de teste
predictions = model.predict(X_test)

# Converter as previsões de one-hot para rótulos originais (materias)
previsoes_classes_test = np.argmax(predictions, axis=1)
previsoes_subjects_test = np.array(materias_cols)[previsoes_classes_test]

# Rótulos reais de volta para os rótulos originais (materias)
true_subjects_test = np.array(materias_cols)[y_test.argmax(axis=1)]

# Adicionar a coluna 'Professor' de volta ao conjunto de testes para criar o DataFrame
X_test_with_professor = X_test.copy()
X_test_with_professor['professor'] = df1['professor'].iloc[X_test.index]

# Imprimir os resultados
results_df = print_results(X_test_with_professor['professor'], previsoes_subjects_test, true_subjects_test)

# Plotar curva de aprendizado
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Curva de Aprendizado')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend()
plt.show()

plt.figure(figsize=(12, 8))
sns.countplot(y='Matéria Real', hue='Matéria Prevista', data=results_df)
plt.title('Comparação entre Matérias Previstas e Reais')
plt.show()

# Criar DataFrame para Plotly Express
plotly_df = results_df.melt(id_vars=['professor'], value_vars=['Matéria Real', 'Matéria Prevista'],
                            var_name='Tipo', value_name='Matéria')

# Gráfico interativo
fig = px.bar(plotly_df, x='professor', y='Matéria', color='Tipo', barmode='group',
             title='Comparação entre Matérias Previstas e Reais')
fig.show()

# Relatório de Desempenho
print(classification_report(true_subjects_test, previsoes_subjects_test))

model.save('modelo_treinado.h5')

model = keras.models.load_model('modelo_treinado.h5')

accuracy = model.evaluate(X_train, y_train, verbose=1)
print(f'Acurácia do Modelo: {accuracy[1]*100:.2f}%')