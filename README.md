# stratdevmodel

 
Phase 1: Trend Detection Model
Develop a machine learning model that identifies trend states using historical crypto price data.

Methodology:
1. **Data Collection & Preprocessing:**
   - Load historical price data from **TradingView CSV exports** (e.g., `BTC.csv`, `TOTAL.csv`, `ETH.csv`)
   - Calculate indicators such as **SMIEO, Trend Magic, and ADF trend filter**
   - Normalize and preprocess the data for AI training

2. **Model Development (PyTorch-based AI)**
   - Implement a **Neural Network (LSTM or Transformer)** trained on **candle closings, rate of change, and rolling mean trends**
   - Utilize **Supervised Learning** by labeling historical data with investor-intended signals
   - Classify the market into **trend states** (positive, negative, neutral)

3. **Training & Evaluation:**
   - Train the model using labeled trend data
   - Validate against unseen test data
   - Optimize the model’s **accuracy in identifying market regimes**

4. **Deployment & Integration:**
   - Export trained AI model
   - Generate **entry (green) and exit (red) signals** on new data
   - Visualize trend states over historical price charts
