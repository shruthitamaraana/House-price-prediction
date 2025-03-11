

---

```markdown
# 🏠 California House Price Prediction

A Streamlit web app to predict median house prices in California using machine learning regression models. Users can input house features via sliders and get instant predictions.



# Features
- User Input: Adjust house features (rooms, bedrooms, income, location, etc.) using sliders.
- Real-Time Prediction: Instantly predicts median house value in USD.
- Model Comparison: Supports Linear Regression, Decision Tree, and Random Forest models.
- Performance Metrics: Displays MAE (Mean Absolute Error) and RMSE (Root Mean Squared Error).

# Dataset
- California Housing Dataset (from `scikit-learn`):
  - Contains block group averages for 8 features (e.g., income, rooms, population).
  - Target variable: Median house value (in $100,000s).

# Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/california-house-price-prediction.git
   cd california-house-price-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

# Usage
1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open `http://localhost:8501` in your browser.

3. Adjust the sliders in the sidebar and view predictions!

# Tech Stack
- **Python**: Primary programming language.
- **Streamlit**: Web app framework.
- **Scikit-learn**: Machine learning models (Linear Regression, Decision Tree, Random Forest).
- **Pandas/NumPy**: Data manipulation.
- **Matplotlib/Seaborn**: Visualization.

# File Structure
```
california-house-price-prediction/
├── app.py               # Main Streamlit application
├── requirements.txt     # Dependencies
├── assets/              # Screenshots/plots
├── README.md            # Documentation
└── data/                # Dataset (loaded via scikit-learn)
```

# Notes
- Predictions are based on **block group averages**, not individual houses.
- For better accuracy, use the Random Forest model (default).

# Acknowledgments
- Dataset: [scikit-learn California Housing Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)
- Built with [Streamlit](https://streamlit.io)

# License
MIT License. See [LICENSE](LICENSE) for details.
```
![Screenshot 2025-03-11 222534](https://github.com/user-attachments/assets/ae68deb6-2aca-42a0-9ccc-b5dc72543dcd)




