# test_load.py
import faiss
import time

path = "/home/a1/sdb/wikidata4rag/Search-r1/e5_Flat.index"
print("Start loading...")
start = time.time()
# 使用 mmap 标志
index = faiss.read_index(path, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
print(f"Loaded in {time.time() - start:.2f}s")
print(f"ntotal: {index.ntotal}")