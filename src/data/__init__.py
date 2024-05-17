import sys
from pathlib import Path

DATAPATH = Path(sys.modules[__name__].__file__).parent

if not DATAPATH.exists():
    DATAPATH = "src/data"
else:
    DATAPATH = str(DATAPATH)