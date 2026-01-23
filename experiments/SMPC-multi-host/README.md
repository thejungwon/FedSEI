# SMPC Multi-host Benchmarks

## Overview

CrypTen multi-party CPU latency benchmarks for mlp/cnn/lenet/resnet18.

## Local multiprocess (single machine)

```
python benchmark_crypten_multi_host_cpu.py --model lenet --batch-size 32 \
  --world-size 3 --launcher mp --runs 5
```

## Multi-host (env launcher)

```
# On each host/terminal, use run_node.sh with unique --rank values
bash run_node.sh --rank 0 --master-addr <master-ip> --master-port 29500 --run-id 0 --log-dir ./logs --quiet -- \
  --model resnet18 --batch-size 1 --mpc-iters 5 --ensemble-k 2

bash run_node.sh --rank 1 --master-addr <master-ip> --master-port 29500 --run-id 0 --log-dir ./logs --quiet -- \
  --model resnet18 --batch-size 1 --mpc-iters 5 --ensemble-k 2

bash run_node.sh --rank 2 --master-addr <master-ip> --master-port 29500 --run-id 0 --log-dir ./logs --quiet -- \
  --model resnet18 --batch-size 1 --mpc-iters 5 --ensemble-k 2
```

## Setup helper (Ubuntu/EC2)

```
bash setup_node.sh
```

## AWS/EC2 notes

- This setup assumes multiple EC2 instances in the same VPC/subnet.
- Open the master port (e.g., 29500) in the security group for intra-VPC traffic.
- Use the private IP of the rank 0 instance as `--master-addr`.
- Run `setup_node.sh` once per instance to install dependencies.
