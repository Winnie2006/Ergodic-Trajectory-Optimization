# Ergodic Trajectory Optimization

This project studies gradient vanishing in ergodic trajectory optimization.

## How to run
python src/main.py --config configs/no_constraint.yaml

## Repository structure
```
src/
│
├── ergodic/
│   ├── metric.py          # 生成 φ_k
│   └── cost.py            # E(φ, q)
│
├── dynamics/
│   └── robot.py
│
├── optimizer/
│   └── ilqr.py
│
├── constraints/
│   ├── collision.py
│   ├── communication.py
│   └── custom_constraint.py   # 你第5步
│
├── system/
│   └── problem.py         # 把所有东西组装起来
│
├── utils/
│   ├── logger.py
│   └── visualization.py
│
└── main.py
```