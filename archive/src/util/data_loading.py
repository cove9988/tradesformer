
def load_data(csv_file):
    data = pd.read_csv(csv_file)


    sequence_length = len(features)  # Number of past observations to consider
    scaler = MinMaxScaler()
    data[features] = scaler.fit_transform(data[features])

    # Check for NaN or Inf values after scaling
    if data[features].isnull().values.any() or np.isinf(data[features].values).any():
        logger.error("Data contains NaN or Inf values after scaling")
        raise ValueError("Data contains NaN or Inf values after scaling")

    # Reset index
    data = data.reset_index()
    logger.info("Data loaded and preprocessed successfully")

    def create_sequences(df, seq_length):
        logger.info("Creating sequences...")
        sequences = []
        for i in range(len(df) - seq_length):
            seq = df.iloc[i:i+seq_length][features].values
            sequences.append(seq)
        logger.info("Sequences created successfully")
        return np.array(sequences)

    sequences = create_sequences(data, sequence_length)
    return data
