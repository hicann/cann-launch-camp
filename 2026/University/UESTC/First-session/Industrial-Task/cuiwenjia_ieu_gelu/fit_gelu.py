"""
Padé rational-sigmoid 系数拟合脚本。

使用 differential evolution + COBYLA 最小化最大绝对误差，
求解 GELU 近似的最优有理分式系数。

公式：
    z = x * (C0 + C2*x²) / (1 + D2*x²)
    GELU(x) = x * sigmoid(2*z) = x / (1 + exp(-2*z))
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.special import erf


def gelu_exact(x):
    return x * 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def gelu_approx(x, C0, C2, D2):
    z = x * (C0 + C2 * x**2) / (1.0 + D2 * x**2)
    return x / (1.0 + np.exp(-2.0 * z))


def max_err(params):
    C0, C2, D2 = params
    x = np.linspace(-8, 8, 100000)
    err = np.max(np.abs(gelu_approx(x, C0, C2, D2) - gelu_exact(x)))
    return err


print("=" * 50)
print("Padé rational-sigmoid 系数拟合")
print("=" * 50)

print("\n[1/2] Differential evolution (Minimax)...")
res = differential_evolution(max_err,
    bounds=[(0.78, 0.81), (0.04, 0.05), (0.009, 0.012)],
    maxiter=200, popsize=30, tol=1e-10, seed=42)
C0_de, C2_de, D2_de = res.x
print(f"  C0={C0_de:.16f}  C2={C2_de:.16f}  D2={D2_de:.16f}")
print(f"  max error: {res.fun:.2e}")

print("\n[2/2] COBYLA refinement...")
res2 = minimize(max_err, [C0_de, C2_de, D2_de], method='COBYLA',
    bounds=[(0.78, 0.81), (0.04, 0.05), (0.009, 0.012)],
    options={'maxiter': 5000, 'tol': 1e-12, 'rhobeg': 1e-4})
C0_f, C2_f, D2_f = res2.x
err_f = max_err([C0_f, C2_f, D2_f])
print(f"  C0={C0_f:.16f}  C2={C2_f:.16f}  D2={D2_f:.16f}")
print(f"  max error: {err_f:.2e}")

print("\n" + "=" * 50)
print("最终系数（可直接用于 kernel）：")
print("=" * 50)
print(f"constexpr float TANH_RATIONAL_C0 = {C0_f:.16f}f;")
print(f"constexpr float TANH_RATIONAL_C2 = {C2_f:.16f}f;")
print(f"constexpr float TANH_RATIONAL_D2 = {D2_f:.16f}f;")
print()
print(f"GELU(x) = x / (1 + exp(-2 * x * ({C0_f:.10f} + {C2_f:.10f}*x²) / (1 + {D2_f:.10f}*x²)))")
print(f"理论最大绝对误差: {err_f:.2e}")
print(f"half 精度门槛 1e-3: {'通过' if err_f < 1e-3 else '不通过'}")
print(f"fp32 精度门槛 1e-4: {'通过' if err_f < 1e-4 else '不通过'}")

# 对比其他近似
x = np.linspace(-8, 8, 200000)


def hendrycks(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


print("\n" + "=" * 50)
print("精度对比")
print("=" * 50)
print(f"{'方法':<35} {'最大误差':<12} {'half':<8} {'fp32':<8}")
print("-" * 63)
for name, func in [
    ("Hendrycks tanh", hendrycks),
    ("Padé rational-sigmoid (本方案)", lambda x: gelu_approx(x, C0_f, C2_f, D2_f)),
]:
    e = np.max(np.abs(func(x) - gelu_exact(x)))
    print(
        f"{name:<35} {e:<12.2e} {'✅' if e < 1e-3 else '❌':<8} {'✅' if e < 1e-4 else '❌':<8}"
    )
