"""Smoke test MVP-1 imports + tiny loads."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path("D:/hoangpc/VS-LeakKG/src").resolve()))

from vsleakkg import load_litpcba_ave, load_dekois, run_mvp1_audit
print("imports OK")

# Tiny AVE load (one small target)
ave_root = Path("D:/hoangpc/VS-LeakKG/data/raw/LIT-PCBA/splits/AVE_unbiased")
df = load_litpcba_ave.load_target(ave_root, "ESR1_ago")
print("ESR1_ago rows:", df.height, "splits:", df["split"].unique().to_list(),
      "labels:", df["label"].unique().to_list())

# DEKOIS extract-and-load smoke
import zipfile, tempfile
zp = Path("D:/hoangpc/VS-LeakKG/data/raw/DEKOIS/DEKOIS2.zip")
# Don't actually extract here — just confirm classifier function
print("DEKOIS classify BDB1:", load_dekois._classify("BDB12345"))
print("DEKOIS classify ZINC1:", load_dekois._classify("ZINC9999"))
print("DEKOIS classify weird:", load_dekois._classify("CHEMBL1"))
print("ALL SMOKE OK")
