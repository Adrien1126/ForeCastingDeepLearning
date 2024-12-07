import matplotlib.pyplot as plt

def plot_close_price(close_price_series, bank_name):
    """
    Trace les prix de clôture ajustés (Adj Close) d'une banque donnée au cours du temps.
    
    Args:
    - close_price_series (pd.Series): Série temporelle des prix de clôture ajustés.
    - bank_name (str): Nom de la banque (pour le titre et la légende).
    
    Returns:
    - None: Affiche le graphique.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(close_price_series.index, close_price_series, label=f'{bank_name} Close Price', color='blue')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title(f'Prix de clôture ajustés de {bank_name} au cours du temps')
    plt.legend()
    plt.show()

def plot_log_return(log_return_series, bank_name):
    """
    Trace le log return d'une banque donnée au cours du temps.
    
    Args:
    - log_return_series (pd.Series): Série temporelle des log returns.
    - bank_name (str): Nom de la banque (pour le titre et la légende).
    
    Returns:
    - None: Affiche le graphique.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(log_return_series.index, log_return_series, label=f'{bank_name} Log Return')
    plt.xlabel('Date')
    plt.ylabel('Log Return')
    plt.title(f'Log Return de {bank_name} au cours du temps')
    plt.legend()
    plt.show()

def plot_learning_curves(history):
    """
    Plots the learning curves for training and validation loss and accuracy.
    
    Args:
    - history: History object returned by model.fit() method.
    """
    # Plotting training & validation loss values
    plt.figure(figsize=(14, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')

    # Accuracy plot (if applicable)
    if 'accuracy' in history.history:
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')
        
    plt.tight_layout()
    plt.show()