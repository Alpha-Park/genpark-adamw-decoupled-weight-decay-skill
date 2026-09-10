import sys
from client import AdamWOptimizer

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

def run():
    print(">>> Demonstrating AdamW Optimizer...")
    opt = AdamWOptimizer(lr=0.05, weight_decay=0.02)
    param = 10.0
    print(f"Initial param value: {param}")

    for step in range(1, 11):
        # Simulated gradient pointing towards 0 (loss = param^2 / 2, grad = param)
        grad = param
        param = opt.step(param, grad)
        if step % 2 == 0:
            print(f"Step {step:2d}: param = {param:.6f}")

    assert param < 1.0
    print("[PASS] AdamW Optimizer converged successfully.")

if __name__ == "__main__":
    run()
