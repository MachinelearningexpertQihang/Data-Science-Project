Please take a look at my code~
Data!!!!!!!! Science!!!!!!!
Project 1
The goal of this project is to build a predictive model for stock prices using an LSTM neural network. LSTMs are a type of Recurrent Neural Network (RNN) that are well-suited for time series data due to their ability to capture long-term dependencies. The model is trained on historical stock prices and can be used to predict future prices.

Model Architecture
The LSTM model consists of the following components:

LSTM Layer: The core of the model, which processes sequential data and captures temporal dependencies.

Fully Connected Layer: A linear layer that maps the LSTM output to the predicted stock price.

Loss Function: Mean Squared Error (MSE) is used to measure the difference between predicted and actual stock prices.

Optimizer: Adam optimizer is used to update the model's weights during training.

Key Parameters:
input_size: Number of features in the input data (1 for univariate time series).

hidden_size: Number of hidden units in the LSTM layer.

num_layers: Number of stacked LSTM layers.

output_size: Number of output features (1 for predicting a single value).

Data Preprocessing
Data Collection: Historical stock data is fetched using the yfinance library.

Normalization: The data is normalized using MinMaxScaler to scale the values between 0 and 1.

Sequence Creation: The data is split into sequences of a fixed length (seq_length) to create input-output pairs for training.

Train-Test Split: The data is split into training and testing sets (80% training, 20% testing).

Training the Model
The model is trained using the following steps:

Forward Pass: The input sequence is passed through the LSTM layer, and the output is generated.

Loss Calculation: The Mean Squared Error (MSE) between the predicted and actual values is computed.

Backward Pass: Gradients are calculated using backpropagation, and the model's weights are updated using the Adam optimizer.

Epochs: The training process is repeated for a fixed number of epochs (e.g., 100).

Backtesting the Model
The model is evaluated on the test set using the following metrics:

Mean Squared Error (MSE): Measures the average squared difference between predicted and actual values.

Root Mean Squared Error (RMSE): The square root of MSE, providing a more interpretable error metric.

Mean Absolute Error (MAE): Measures the average absolute difference between predicted and actual values.

The results are visualized by plotting the predicted and actual stock prices.

Future Predictions
The trained model can be used to predict future stock prices. Given the last known sequence of stock prices, the model predicts the next n time steps (e.g., 30 days). The predictions are inverse-transformed to their original scale and visualized.



Project2:Game Behavior Data Collection SDK (GameEventTracker Class) This part implements the in-game event tracking system. The main functions include:
Initialize player sessions and user information Track various game events (startup, payment, upgrade, task completion, card drawing, etc.) Ensure data security through a signature mechanism Send data to the backend server for analysis

This SDK is designed to be embedded in the game client to collect player behavior data and provide a basic data source for subsequent analysis. 2. Player Churn Warning Model (ChurnPredictionModel Class) This is a machine learning model used to predict the risk of player churn:

Prepare multi-dimensional feature data, including activity, game progress, social behavior, and consumption habits Use the random forest algorithm to train the churn prediction model Evaluate model performance and analyze feature importance Predict the probability of churn for each player and classify it as low/medium/high risk Support model saving and loading

This model helps the operation team identify players who may churn in advance and implement targeted retention measures. 3. Real-time data processing Flink task (process_game_events function) This part processes real-time game data streams, and its functions include:

Use Apache Flink framework to process event streams in Kafka Set up a checkpoint mechanism to ensure fault tolerance Validate, transform and enrich raw event data Filter invalid events to ensure data quality Send processed data to downstream systems

This module implements the basic work of real-time data analysis and supports real-time monitoring and rapid response. 4. Data visualization Dashboard (Dash application) This is an interactive data analysis platform that mainly displays:

Key business indicator (KPI) panel: DAU, revenue, new users, retention rate, etc. Trend chart: daily active user changes, revenue trends User retention heat map: display user retention in different periods Player level distribution and payment conversion funnel chart Support filtering data by game and time range

The platform provides decision makers with intuitive data visualization and supports data-driven decision making.
