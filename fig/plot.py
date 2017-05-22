import numpy as np
import matplotlib.pyplot as plt



p = [1, 2, 4, 8, 16, 32, 64, 128]

# Weak node
node_weak_time = [1.07, 2.14, 4.77, 9.97, 18.31, 31.62, 71.63, 235.46]

# Strong node
node_strong_time = [49.66, 35.33, 31.96, 31.85, 32.06, 32.05, 32.42, 32.37]
# plt.savefig("strong_scaling-node.png")

# Weak feature

feat_weak_time = [3.1243, 3.3422, 3.6725, 3.9702, 5.4244, 8.2892, 12.7914, 25.7922]

# Strong feature

feat_strong_time = [44.4568, 22.6877, 11.7013, 6.1795, 3.9522, 2.8689, 2.2273, 2.1015]

base_node = 49.66
base_feat = 44.4568

speed_node = [base_node/x for x in node_strong_time]
speed_feature = [base_feat/x for x in feat_strong_time]

# optimal = [1, 2, 4, 8, 16, 32, 64, 128]




plt.xticks(p)
plt.ylim(0, 30)
# plt.yticks(range(0, 100))

# plt.yticks(np.arange(40, 73))

# plt.le
# plt.plot(p, optimal, 'r', label = 'Optimal')
plt.plot(p, feat_weak_time, 'g', label = "Parallel Feature")
# plt.plot(p, node_strong_time, 'b', label = "Parallel Node")
plt.xlabel('# of Threads')
# plt.ylabel('Time (second)')
plt.ylabel('Time')
plt.legend()

plt.savefig("weak-feat.png")
# plt.show()

