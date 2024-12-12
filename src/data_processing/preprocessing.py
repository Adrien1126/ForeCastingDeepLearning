import yfinance as yf
import pandas as pd
import numpy as np

def prepare_stock_data(symbols, start_date, end_date):
    """
    Télécharge et prépare les données boursières pour une liste de symboles.
    
    Args:
    - symbols (list): Liste des symboles des actions.
    - start_date (str): Date de début au format 'YYYY-MM-DD'.
    - end_date (str): Date de fin au format 'YYYY-MM-DD'.
    
    Returns:
    - pd.DataFrame: Un DataFrame combiné avec les données traitées.
    """
    data = {}

    # Télécharger les données pour chaque symbole
    for ticker in symbols:
        print(f"Téléchargement des données pour {ticker}...")
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        data[ticker] = stock_data

    # Combiner les données de toutes les actions en un seul DataFrame
    combined_data = pd.concat(data, axis=1)
    
    # Compter les valeurs manquantes initiales
    first_non_nan_index = combined_data.apply(lambda col: col.first_valid_index())
    num_missing_start = combined_data.index.get_loc(first_non_nan_index.min())
    print(f"Nombre de valeurs manquantes au début : {num_missing_start}")
    
    # 1. Interpolation pour les valeurs manquantes intermédiaires
    combined_data = combined_data.interpolate(method='linear')

    # 2. Forward fill pour les valeurs manquantes au début
    combined_data = combined_data.ffill()

    # 3. Backward fill pour les valeurs manquantes à la fin
    combined_data = combined_data.bfill()

    # Vérification finale
    missing_values = combined_data.isnull().sum().sum()
    print(f"Nombre total de valeurs manquantes après traitement: {missing_values}")
    
    return combined_data


def add_log_returns_and_volume(data, symbols):
    """
    Calcule les log returns et conserve toutes les colonnes d'origine dans un DataFrame combiné.
    
    Args:
    - combined_data (pd.DataFrame): DataFrame combiné contenant les données des actions.
    - symbols (list): Liste des symboles des actions.
    
    Returns:
    - pd.DataFrame: DataFrame avec les colonnes d'origine, les log returns et les volumes par action.
    """
    # Créer une copie du DataFrame d'origine pour préserver toutes les colonnes
    enhanced_data = data.copy()
    
    # Calculer les log returns et ajouter les volumes
    for ticker in symbols:
        print(f"Calcul des log returns et du volume pour {ticker}...")
        try:
            # Calcul du log return
            enhanced_data[(ticker, 'Log Return')] = np.log(
                data[(ticker, 'Adj Close')] / data[(ticker, 'Adj Close')].shift(1)
            )
            # Ajouter les volumes (si déjà présents, ils resteront inchangés)
            if (ticker, 'Volume') not in enhanced_data.columns:
                enhanced_data[(ticker, 'Volume')] = data[(ticker, 'Volume')]
        except KeyError:
            print(f"Données manquantes pour {ticker}. Vérifiez le DataFrame source.")
    
    # Supprimer les lignes contenant des valeurs NaN résultant des calculs
    enhanced_data = enhanced_data.dropna()
    
    # Afficher un aperçu des données finales
    print("Aperçu des données avec log returns et colonnes conservées :")
    print(enhanced_data.head())
    print(f"Taille du DataFrame amélioré : {enhanced_data.shape}")
    
    return enhanced_data

def add_moving_averages(data, symbols, windows=[10, 50, 200]):
    """
    Calcule les moyennes mobiles et conserve toutes les colonnes d'origine dans un DataFrame combiné.

    Args:
    - data (pd.DataFrame): DataFrame combiné contenant les données des actions.
    - symbols (list): Liste des symboles des actions.
    - windows (list): Liste des périodes pour lesquelles calculer les moyennes mobiles (par défaut : [10, 50, 200]).

    Returns:
    - pd.DataFrame: DataFrame avec les colonnes d'origine et les moyennes mobiles ajoutées.
    """
    # Créer une copie du DataFrame d'origine pour préserver toutes les colonnes
    enhanced_data = data.copy()

    # Calculer les moyennes mobiles pour chaque symbole et chaque fenêtre
    for ticker in symbols:
        print(f"Calcul des moyennes mobiles pour {ticker}...")
        try:
            for window in windows:
                # Ajouter la moyenne mobile avec une nouvelle colonne
                enhanced_data[(ticker, f'SMA_{window}')] = (
                    data[(ticker, 'Adj Close')].rolling(window=window).mean()
                )
        except KeyError:
            print(f"Données manquantes pour {ticker}. Vérifiez le DataFrame source.")

    # Supprimer les lignes contenant des NaN résultant des calculs de rolling
    enhanced_data = enhanced_data.dropna()

    # Afficher un aperçu des données finales
    print("Aperçu des données avec moyennes mobiles et colonnes conservées :")
    print(enhanced_data.head())
    print(f"Taille du DataFrame amélioré : {enhanced_data.shape}")

    return enhanced_data


def remove_high_and_open_and_low(data, symbols):
    """
    Supprime les colonnes 'High' et 'Open' tout en conservant toutes les colonnes restantes.

    Args:
    - data (pd.DataFrame): DataFrame combiné contenant les données des actions.
    - symbols (list): Liste des symboles des actions.

    Returns:
    - pd.DataFrame: DataFrame sans les colonnes 'High' et 'Open', avec toutes les autres colonnes conservées.
    """
    # Créer une copie du DataFrame d'origine pour préserver toutes les colonnes restantes
    cleaned_data = data.copy()

    # Supprimer les colonnes 'High' et 'Open' pour chaque ticker
    for ticker in symbols:
        print(f"Suppression des colonnes 'High' et 'Open' pour {ticker}...")
        try:
            # Supprimer les colonnes 'High' et 'Open' si elles existent
            columns_to_remove = [(ticker, 'High'), (ticker, 'Open'), (ticker, 'Low')]
            columns_to_remove = [col for col in columns_to_remove if col in cleaned_data.columns]
            cleaned_data = cleaned_data.drop(columns=columns_to_remove, axis=1)
        except KeyError:
            print(f"Colonnes manquantes pour {ticker}. Aucune suppression nécessaire.")

    # Afficher un aperçu des données finales
    print("Aperçu des données après suppression des colonnes 'High' et 'Open' :")
    print(cleaned_data.head())
    print(f"Taille du DataFrame nettoyé : {cleaned_data.shape}")

    return cleaned_data

def data_splitting(data,train_size,test_size,validation_size,gap):

    # Calculate indices
    train_end = train_size
    test_start = train_end + gap
    test_end = test_start + test_size
    validation_start = test_end + gap
    validation_end = validation_start + validation_size
    if validation_end > len(data):
        raise ValueError("Dataset size is too small for the specified splits and gap.")
    
    # Create subsets
    train_indices = np.arange(0, train_end)
    test_indices = np.arange(test_start, test_end)
    validation_indices = np.arange(validation_start, validation_end)

    train = data.iloc[train_indices]
    test = data.iloc[test_indices]
    validation = data.iloc[validation_indices]

    return train, test, validation

# Function to add moving averages and other engineered features
def add_features(data, window_sizes=[5, 10, 20]):
    """
    Add moving averages and other features to the dataset.

    :param data: DataFrame containing the stock data with at least 'Adj Close' and 'Volume'.
    :param window_sizes: List of integers representing window sizes for moving averages.
    :return: DataFrame with added features.
    """
    for window in window_sizes:
        # Moving Average for Adjusted Close Price
        data[f'ma_close_{window}'] = data['Adj Close'].rolling(window=window).mean()
        # Moving Average for Volume
        data[f'ma_volume_{window}'] = data['Volume'].rolling(window=window).mean()

    # Relative Strength Index (RSI)
    delta = data['Adj Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    for window in window_sizes:
        rolling_mean = data['Adj Close'].rolling(window=window).mean()
        rolling_std = data['Adj Close'].rolling(window=window).std()
        data[f'bollinger_upper_{window}'] = rolling_mean + (2 * rolling_std)
        data[f'bollinger_lower_{window}'] = rolling_mean - (2 * rolling_std)

    # Missing values
    data=data.infer_objects(copy=False)
    data=data.ffill()
    data=data.bfill()

    return data