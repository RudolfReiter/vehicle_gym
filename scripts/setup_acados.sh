#!/usr/bin/env bash
# A local, pinned acados build. No changes to a system installation are needed.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
acados_dir="${ACADOS_SOURCE_DIR:-$repo_dir/artifacts/acados-source}"
revision="fb0485a8e948589425f15dc652b813d12b339c33"
if [[ ! -d "$acados_dir/.git" ]]; then
  git clone https://github.com/acados/acados.git "$acados_dir"
fi
if [[ -n "$(git -C "$acados_dir" status --porcelain --untracked-files=no)" ]]; then
  printf 'The acados checkout has tracked changes: %s\n' "$acados_dir" >&2
  exit 1
fi
git -C "$acados_dir" checkout "$revision"
git -C "$acados_dir" submodule update --init --recursive
cmake -S "$acados_dir" -B "$acados_dir/build" \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_INSTALL_PREFIX="$acados_dir" \
  -DBLASFEO_TARGET=GENERIC -DHPIPM_TARGET=GENERIC \
  -DBLASFEO_EXAMPLES=OFF -DACADOS_WITH_OPENMP=OFF \
  -DCMAKE_SHARED_LINKER_FLAGS=-Wl,-z,noexecstack
cmake --build "$acados_dir/build" --parallel 4
cmake --install "$acados_dir/build"
python -m pip install future-fstrings -e "$acados_dir/interfaces/acados_template"
ACADOS_SOURCE_DIR="$acados_dir" python - <<'PY'
import os
from pathlib import Path
import sys
import urllib.request
from acados_template.utils import TERA_VERSION, PLATFORM2TERA, get_tera_exec_path
path=Path(get_tera_exec_path())
if not path.is_file():
    url=f"https://github.com/acados/tera_renderer/releases/download/v{TERA_VERSION}/t_renderer-v{TERA_VERSION}-{PLATFORM2TERA[sys.platform]}"
    path.parent.mkdir(parents=True,exist_ok=True)
    urllib.request.urlretrieve(url,path)
    path.chmod(0o755)
print(f"Template renderer: {path}")
PY
printf '\nExport these variables before running races:\n'
printf 'export ACADOS_SOURCE_DIR=%q\n' "$acados_dir"
printf 'export LD_LIBRARY_PATH=%q${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\n' "$acados_dir/lib"
printf 'export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1\n'
