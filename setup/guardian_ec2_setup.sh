#!/bin/bash
# EC2 setup script for Guardian XML extraction pipeline.
# Run once on a fresh Amazon Linux 2023 / Ubuntu instance.
# Assumes the instance role has s3:GetObject on raw/guardian/ and
# s3:PutObject on interim/guardian/.

set -euo pipefail

echo "=== Installing system dependencies ==="
if command -v dnf &>/dev/null; then
    # Amazon Linux 2023
    sudo dnf install -y python3.11 python3.11-pip python3.11-devel git
    PYTHON=python3.11
    PIP=pip3.11
elif command -v apt-get &>/dev/null; then
    # Ubuntu
    sudo apt-get update -qq
    sudo apt-get install -y python3.11 python3.11-pip python3.11-venv git
    PYTHON=python3.11
    PIP=pip3.11
else
    echo "ERROR: unsupported package manager"
    exit 1
fi

echo "=== Creating virtualenv ==="
$PYTHON -m venv /opt/idiom_index_venv
source /opt/idiom_index_venv/bin/activate

echo "=== Installing Python dependencies ==="
pip install --upgrade pip
pip install \
    nltk \
    pandas \
    pyarrow \
    pyyaml \
    boto3 \
    tqdm

echo "=== Downloading NLTK data ==="
python -c "
import nltk
nltk.download('punkt', quiet=False)
nltk.download('punkt_tab', quiet=False)
"

echo "=== Cloning / copying project ==="
# Option A: clone from git (replace with your repo URL)
# git clone https://github.com/YOUR_ORG/idiom-index.git /opt/idiom_index
#
# Option B: copy from S3 (if you've uploaded the project there)
# aws s3 cp s3://idiom-index/project.tar.gz /tmp/project.tar.gz
# tar -xzf /tmp/project.tar.gz -C /opt/idiom_index
#
# For now, assume the project is already at /opt/idiom_index
echo "NOTE: copy or clone the project to /opt/idiom_index before running extraction."

echo "=== Writing run script ==="
cat > /opt/run_guardian_extract.sh << 'RUNSCRIPT'
#!/bin/bash
set -euo pipefail
source /opt/idiom_index_venv/bin/activate
cd /opt/idiom_index/project

echo "Starting Guardian extraction at $(date)"
python src/01d_extract_guardian.py \
    --s3-input  s3://idiom-index/raw/guardian/ \
    --s3-output s3://idiom-index/interim/guardian/ \
    --workers   $(nproc) \
    --log-level INFO \
    2>&1 | tee /var/log/guardian_extract_$(date +%Y%m%d_%H%M%S).log

echo "Done at $(date)"
RUNSCRIPT

chmod +x /opt/run_guardian_extract.sh

echo ""
echo "=== Setup complete ==="
echo "To run extraction:"
echo "  /opt/run_guardian_extract.sh"
echo ""
echo "To run a test on first 5 files:"
echo "  source /opt/idiom_index_venv/bin/activate"
echo "  cd /opt/idiom_index/project"
echo "  python src/01d_extract_guardian.py \\"
echo "      --s3-input  s3://idiom-index/raw/guardian/ \\"
echo "      --s3-output s3://idiom-index/interim/guardian/ \\"
echo "      --limit 5 --log-level DEBUG"
