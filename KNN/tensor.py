import numpy as np
import faiss
import matplotlib.pyplot as plt
import torch
import time
import faiss.contrib.torch_utils
data_point = np.random.randn(2000, 512)
data_norm = np.linalg.norm(data_point, axis=1, keepdims=True)
ftrain = data_point / data_norm
ftrain = np.ascontiguousarray(ftrain.astype(np.float32))
id_train_size = 1000
K = 50
res = faiss.StandardGpuResources()
rand_ind = np.random.choice(id_train_size, int(id_train_size), replace=False)
ftrain = torch.tensor(ftrain).cuda()
index = faiss.IndexFlatL2(ftrain.shape[1])
gpu_index = faiss.index_cpu_to_all_gpus(index, ngpu=8)
for i in range(0):
    gpu_index.add(ftrain)
    start_time = time.time()
D, idx = gpu_index.search(ftrain, K)
end_time = time.time()

print("time cost:", float(end_time - start_time) * 1000.0, "ms")
scores_in = D[:, -1].cpu()
top_idx = np.argsort(-scores_in)[:1]
top_2_idx = np.argsort(scores_in)[:1]
point_time = data_point[top_idx]
point_top_2 = data_point[top_2_idx]
ftrain_T = data_point.T
point_time_T = point_time.T
point_top_2 = point_top_2.T
#plt.scatter(ftrain_T[0], ftrain_T[1])
#plt.scatter(point_time_T[0], point_time_T[1], c='g')
#plt.scatter(point_top_2[0], point_top_2[1], c='r')
#plt.savefig('/afs/cs.wisc.edu/u/t/a/taoleitian/t_2_9.png')
#plt.show()