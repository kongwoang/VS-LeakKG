import sys, importlib.util
print("python", sys.version.splitlines()[0])
print("executable", sys.executable)
for m in ["rdkit","polars","pyarrow","pandas","numpy","scipy","sklearn","networkx","tqdm","rich","Bio"]:
    spec = importlib.util.find_spec(m)
    print(m, "ok" if spec else "MISSING")
