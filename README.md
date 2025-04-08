# Linear Regression from Scratch

This project implements a simple linear regression model in Python using gradient descent. The model is trained to predict car prices based on mileage.

---

## 📂 Files Overview

- `train_model.py` – Trains the linear regression model using gradient descent.
- `predict_price.py` – Uses the trained model to estimate the price of a car given its mileage.
- `data.csv` – Input dataset containing car mileage and price.
- `theta.csv` – Stores the learned parameters (theta0 and theta1) after training.

---

## 📊 Data Format

The dataset `data.csv` should be structured as follows:

```
km,price
240000,3650
139800,3800
150500,4400
...
```

---

## 🔁 Normalization

Before training, the input data is normalized using min-max scaling:

- **Normalized km**:  
  \[
  x_{norm} = \frac{x - \min(x)}{\max(x) - \min(x)}
  \]
- **Normalized price**:  
  \[
  y_{norm} = \frac{y - \min(y)}{\max(y) - \min(y)}
  \]

---

## ⚙️ Training: Gradient Descent

The model minimizes the Mean Squared Error (MSE) using gradient descent:

- Hypothesis function:  
  \[
  \hat{y} = \theta_0 + \theta_1 \cdot x
  \]

- Gradient descent update rules:
  \[
  \theta_0 := \theta_0 - \alpha \cdot \frac{1}{m} \sum (\hat{y}_i - y_i)
  \]
  \[
  \theta_1 := \theta_1 - \alpha \cdot \frac{1}{m} \sum ((\hat{y}_i - y_i) \cdot x_i)
  \]

Where:
- \( m \) is the number of samples
- \( \alpha \) is the learning rate

---

## ✅ Accuracy Metrics

During training, the model evaluates the following metrics:

- **R² Score** (Coefficient of Determination):  
  \[
  R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
  \]

- **MSE** and **MAE** (not printed but easy to add if needed)

---

## 📈 Visualizations

After training, a matplotlib window allows you to slide through all iterations and see how the regression line improves step by step.

---

## 🧠 Predicting Price

Once trained, you can predict the price of a car using:

```bash
python3 predict_price.py <mileage>
```

Example:

```bash
python3 predict_price.py 50000
```

> This uses the `theta.csv` file that stores the trained weights.

---

## 🔧 How to Use

1. **Add your dataset** to `data.csv`
2. **Run training**:
    ```bash
    python3 train_model.py
    ```
3. **Predict price**:
    ```bash
    python3 predict_price.py 100000
    ```

---

## 📎 Notes

- The model uses **min-max normalization**.
- You must **train the model** at least once before running a prediction.
- All parameters are saved in `theta.csv`.

---

## 🧪 Requirements

- Python 3.x
- pandas
- numpy
- matplotlib

Install requirements with:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install pandas numpy matplotlib
```

---

## 🧼 Clean Code

Includes robust error handling and validations for:

- File existence
- Column formatting
- Data consistency and types


---

## 🧮 Math Symbols Used

- \( \cdot \) — Dot product or multiplication
- \( \hat{y} \) — Predicted value (read as "y hat")
- \( \bar{y} \) — Mean of actual values
- \( \alpha \) — Learning rate
- \( m \) — Number of training samples


---

Happy coding! 🚗📉