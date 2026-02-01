#!/usr/bin/env bash
#
# EC2 Ubuntu 초기 셋업 스크립트 (Conda + Python 3.9 + CrypTen 0.4.1)
#

set -euo pipefail

ENV_NAME="${ENV_NAME:-fedsei}"
PYTHON_VERSION="${PYTHON_VERSION:-3.9}"
MINICONDA_DIR="$HOME/miniconda"

echo "==== [1] APT 업데이트 ===="
sudo apt-get update -y
sudo apt-get upgrade -y
sudo apt-get install -y \
  build-essential \
  git \
  curl \
  wget \
  tmux \
  htop

echo
echo "==== [2] Miniconda 설치 ===="
if [ -d "${MINICONDA_DIR}" ]; then
  echo "Miniconda already installed."
else
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
  bash miniconda.sh -b -p "${MINICONDA_DIR}"
  rm miniconda.sh
fi

echo
echo "==== [3] conda 초기화 ===="
eval "$(${MINICONDA_DIR}/bin/conda shell.bash hook)"

# 💡 신규 conda에서 필요
echo
echo "==== [4] conda ToS 자동 동의 ===="
conda tos accept || true

echo
echo "==== [5] Conda env 생성 ===="
if conda env list | grep -q "^${ENV_NAME}\s"; then
  echo "Conda env '${ENV_NAME}' already exists."
else
  conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}"
fi

conda activate "${ENV_NAME}"

echo
echo "==== [6] pip 업그레이드 ===="
pip install "pip<24.1" setuptools wheel

echo
echo "==== [7] Torch 1.8.1 + CPU torchvision 설치 ===="
pip install \
  torch==1.8.1+cpu \
  torchvision==0.9.1+cpu \
  -f https://download.pytorch.org/whl/torch_stable.html

echo
echo "==== [8] Crypten 및 너의 환경 버전들 설치 ===="

export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

pip install \
  crypten==0.4.1 \
  absl-py==2.3.1 \
  future==1.0.0 \
  grpcio==1.76.0 \
  importlib_metadata==8.7.0 \
  Markdown==3.9 \
  MarkupSafe==3.0.3 \
  numpy==1.19.5 \
  omegaconf==2.0.6 \
  onnx==1.10.0 \
  packaging==25.0 \
  pandas==1.2.2 \
  pillow==11.3.0 \
  protobuf==3.20.3 \
  python-dateutil==2.9.0.post0 \
  pytz==2025.2 \
  PyYAML==5.3.1 \
  scipy==1.6.0 \
  six==1.17.0 \
  sklearn==0.0.post12 \
  tensorboard==2.20.0 \
  tensorboard-data-server==0.7.2 \
  typing_extensions==4.15.0 \
  Werkzeug==3.1.3 \
  zipp==3.23.0

echo
echo "==== [9] ~/.bashrc 자동 conda init 추가 ===="

if ! grep -q "## FedSEI conda init" "$HOME/.bashrc"; then
  {
    echo ""
    echo "## FedSEI conda init"
    echo "eval \"\$(${MINICONDA_DIR}/bin/conda shell.bash hook)\""
    echo "conda activate ${ENV_NAME}"
  } >> "$HOME/.bashrc"
fi

echo
echo "==== [10] OMP/MKL 제한 ===="

if ! grep -q "## FedSEI OMP settings" "$HOME/.bashrc"; then
  {
    echo ""
    echo "## FedSEI OMP settings"
    echo "export OMP_NUM_THREADS=1"
    echo "export MKL_NUM_THREADS=1"
  } >> "$HOME/.bashrc"
fi

echo
echo "==== [11] 설치 확인 ===="
python - << 'EOF'
import torch, crypten
print("PyTorch:", torch.__version__)
print("CrypTen:", crypten.__version__)
EOF

echo
echo "==== ✅ FedSEI 환경 준비 완료 ===="
echo "재접속 후 자동 활성화됨"
