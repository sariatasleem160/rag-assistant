import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
a = model.encode("The cat sat on the mat.")
b = model.encode("A kitten rested on the rug.")
c = model.encode("Quarterly revenue grew 12%.")

cos = lambda x, y: np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
print("vector length:", len(a))
print("cat vs kitten:", round(cos(a, b), 3))
print("cat vs revenue:", round(cos(a, c), 3))
