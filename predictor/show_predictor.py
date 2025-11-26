import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 设置英文字体避免警告
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def train_linear_layer_model():
    """使用线性回归训练线性层时延预测模型"""

    # 加载数据
    data = pd.read_csv("./dataset/edge/linear_lat.csv")
    print("Data Information:")
    print(f"Data shape: {data.shape}")
    print(f"First 5 rows:")
    print(data.head())

    # 计算计算量特征 (线性层的FLOPs = 2 * in_features * out_features)
    data['computation'] = 2 * data['in_features'] * data['out_features']

    # 使用计算量作为唯一特征
    X = data[['computation']].values
    y = data['latency'].values

    print(f"\nComputation stats: min={data['computation'].min():.0f}, max={data['computation'].max():.0f}")
    print(f"Latency stats: min={data['latency'].min():.3f}ms, max={data['latency'].max():.3f}ms")
    print(f"Input features range: {data['in_features'].min()}-{data['in_features'].max()}")
    print(f"Output features range: {data['out_features'].min()}-{data['out_features'].max()}")

    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    # 训练线性回归模型（强制截距为0，避免负预测）
    model = LinearRegression(fit_intercept=False)  # 设置截距为0
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)

    # 计算评估指标
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\n=== Linear Model Performance ===")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f} ms")
    print(f"Root Mean Square Error (RMSE): {rmse:.4f} ms")
    print(f"Model coefficient: {model.coef_[0]:.10f}")
    print(f"Intercept: {model.intercept_:.6f}")

    # 保存模型
    model_path = "./config/edge/linear_linear_computation.pkl"
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")

    return model, data, X_test, y_test, y_pred


def plot_linear_layer_model(model, data, X_test, y_test, y_pred):
    """Plot linear layer model fitting results"""

    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Figure 1: Computation vs Latency scatter plot and fit line
    computation_range = np.linspace(data['computation'].min(),
                                    data['computation'].max(), 100)
    latency_pred_range = model.predict(computation_range.reshape(-1, 1))

    # Use linear coordinates
    ax1.scatter(data['computation'], data['latency'], alpha=0.6, s=30, color='blue', label='True Data')
    ax1.plot(computation_range, latency_pred_range, 'r-', linewidth=2, label='Linear Fit')
    ax1.set_xlabel('Computation (FLOPs)')
    ax1.set_ylabel('Latency (ms)')
    ax1.set_title('Computation vs Latency (Linear Regression)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Figure 2: Predicted vs True values
    ax2.scatter(y_test, y_pred, alpha=0.6, s=30, color='green')
    max_val = max(y_test.max(), y_pred.max())
    min_val = min(y_test.min(), y_pred.min())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Ideal Prediction')
    ax2.set_xlabel('True Latency (ms)')
    ax2.set_ylabel('Predicted Latency (ms)')
    ax2.set_title(f'Predicted vs True Latency (R² = {r2_score(y_test, y_pred):.4f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Figure 3: Heatmap of input vs output features vs latency
    # Create grid data for smaller ranges to avoid memory issues
    in_sample = data['in_features'].unique()[:20]  # Sample first 20 unique values
    out_sample = data['out_features'].unique()[:20]
    X_grid, Y_grid = np.meshgrid(in_sample, out_sample)

    # Calculate predicted latency for each grid point
    Z_pred = np.zeros_like(X_grid, dtype=float)
    for i in range(X_grid.shape[0]):
        for j in range(X_grid.shape[1]):
            computation = 2 * X_grid[i, j] * Y_grid[i, j]
            Z_pred[i, j] = max(model.predict([[computation]])[0], 0)  # Ensure non-negative

    contour = ax3.contourf(X_grid, Y_grid, Z_pred, levels=20, cmap='viridis')
    ax3.set_xlabel('Input Features (in_features)')
    ax3.set_ylabel('Output Features (out_features)')
    ax3.set_title('Input vs Output Features vs Predicted Latency')
    plt.colorbar(contour, ax=ax3, label='Predicted Latency (ms)')

    plt.tight_layout()
    plt.show()

    # Show residual plot
    plt.figure(figsize=(10, 6))
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.6, s=30, color='purple')
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.8, label='Zero Error Line')
    plt.xlabel('Predicted Latency (ms)')
    plt.ylabel('Residuals (True - Predicted)')
    plt.title('Residual Plot')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# Prediction function
def predict_linear_latency(in_features, out_features, model_path="./config/edge/linear_linear_computation.pkl"):
    """Predict linear layer latency using linear model"""

    # Load model
    model = joblib.load(model_path)

    # Calculate computation (linear layer FLOPs = 2 * in_features * out_features)
    computation = 2 * in_features * out_features

    # Predict
    features = np.array([[computation]])
    prediction = max(model.predict(features)[0], 0)  # Ensure non-negative

    return prediction, computation


if __name__ == "__main__":
    # Train linear model
    model, data, X_test, y_test, y_pred = train_linear_layer_model()

    # Plot charts
    plot_linear_layer_model(model, data, X_test, y_test, y_pred)

    # Test predictions
    print("\n=== Model Prediction Test ===")
    test_cases = [
        (10, 100),  # Small scale
        (100, 500),  # Medium scale
        (500, 1000),  # Large scale
        (1000, 2000),  # Very large scale
        (10, 10),  # Minimum from dataset
        (8110, 8110),  # Maximum from dataset
    ]

    for i, (in_feat, out_feat) in enumerate(test_cases):
        pred_latency, computation = predict_linear_latency(in_feat, out_feat)

        print(f"Test case {i + 1}:")
        print(f"  Input features: {in_feat}, Output features: {out_feat}")
        print(f"  Computation (FLOPs): {computation:,}")
        print(f"  Predicted latency: {pred_latency:.3f} ms")
        print()

    # Analyze model performance on dataset
    print("\n=== Dataset Performance Analysis ===")
    data['predicted_latency'] = model.predict(data[['computation']].values)
    data['predicted_latency'] = data['predicted_latency'].clip(lower=0)  # Ensure non-negative
    data['error'] = abs(data['latency'] - data['predicted_latency'])
    data['error_percent'] = (data['error'] / data['latency']) * 100

    print(f"Average error: {data['error'].mean():.4f} ms")
    print(f"Maximum error: {data['error'].max():.4f} ms")
    print(f"Average error percentage: {data['error_percent'].mean():.2f}%")

    # Show some examples of predictions vs actual
    print(f"\nSample predictions vs actual:")
    sample_data = data.sample(5)
    for _, row in sample_data.iterrows():
        print(
            f"  FLOPs: {row['computation']:>8,} -> Actual: {row['latency']:6.3f}ms, Pred: {row['predicted_latency']:6.3f}ms, Error: {row['error']:5.3f}ms")