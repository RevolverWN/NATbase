import torch
import pickle


# with open("./en-de_40000_shared.pt", 'rb') as f:
#     content = f.read()
#     res = pickle.loads(content)
res = torch.load("./en-de_40000_shared.pt")
print(res)
# print(res)