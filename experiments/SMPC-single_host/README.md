# SMPC Single-host Benchmarks

Single-machine CrypTen benchmarks (CPU-focused).

Core scripts:

- `benchmark_crypten_parties_cpu.py` (cnn/mlp, multi-party vs plaintext)
- `benchmark_crypten_cnn.py` (small/medium/large CNN, single-process CrypTen)
- `benchmark_crypten_mlp.py` (small/medium/large MLP, single-process CrypTen)

Examples (per file):

benchmark_crypten_parties_cpu.py

# CNN, 3-party MPC

```
python benchmark_crypten_parties_cpu.py --model cnn --batch-size 32 --world-size 3
```

# MLP, 1-party (overhead only)

```
python benchmark_crypten_parties_cpu.py --model mlp --batch-size 32 --world-size 1
```

# Sweep over world sizes and batch sizes

```
python benchmark_crypten_parties_cpu.py --sweep
```

benchmark_crypten_cnn.py

# Small CNN single run

```
python benchmark_crypten_cnn.py --model small --batch-size 32 --device cpu
```

# Full sweep (small/medium/large x batch sizes)

```
python benchmark_crypten_cnn.py --sweep --device cpu
```

benchmark_crypten_mlp.py

# Medium MLP single run

```
python benchmark_crypten_mlp.py --model medium --batch-size 32 --device cpu
```

# Full sweep (small/medium/large x batch sizes)

```
python benchmark_crypten_mlp.py --sweep --device cpu
```

Docker (repeatable run):

# From this folder

```
docker compose up --build
```

Customize via environment variables:

# Parties benchmark with CNN

```
SCRIPT=benchmark_crypten_parties_cpu.py MODEL=cnn \
 BATCH_SIZE=32 WORLD_SIZE=3 EXTRA_ARGS="--repeats 3" \
 docker compose up --build
```

# Parties benchmark with MLP

```
SCRIPT=benchmark_crypten_parties_cpu.py MODEL=cnn \
 BATCH_SIZE=64 WORLD_SIZE=2 EXTRA_ARGS="--plain-iters 20 --mpc-iters 5" \
 docker compose up --build
```

# Single-process CNN sweep

```
SCRIPT=benchmark_crypten_cnn.py MODEL=small \
 EXTRA_ARGS="--device cpu --sweep" \
 docker compose up --build
```

```
SCRIPT=benchmark_crypten_mlp.py MODEL=small \
 BATCH_SIZE=16 EXTRA_ARGS="--device cpu --sweep" \
 docker compose up --build
```
