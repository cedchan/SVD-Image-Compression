import matplotlib.image as image
import matplotlib.pyplot as plt
import numpy as np
import math


img = image.imread('pic.jpeg')

# Make entries between 0 and 1
img = img / 0xff

R = img[:, :, 0]
G = img[:, :, 1]
B = img[:, :, 2]

r = np.linalg.matrix_rank(R)

U_R, S_R, V_R = np.linalg.svd(R)
U_G, S_G, V_G = np.linalg.svd(G)
U_B, S_B, V_B = np.linalg.svd(B)


def reconstruct(U, S, V, k):
    return U[:, :k] @ np.diag(S[:k]) @ V[:k, :]


def compress(k):
    r = reconstruct(U_R, S_R, V_R, k)
    g = reconstruct(U_G, S_G, V_G, k)
    b = reconstruct(U_B, S_B, V_B, k)

    compressed = np.stack((r, g, b), axis=2)
    compressed = compressed * 0xff
    return compressed.astype(np.int32)


def plot_comp(p):
    k = math.ceil(r * (1 - p))
    plt.imshow(compress(k))
    plt.title(f'{(p * 100):.1f}% Compressed')


plt.figure()
plt.subplot(2, 3, 1)
plot_comp(0)
plt.subplot(2, 3, 2)
plot_comp(0.2)
plt.subplot(2, 3, 3)
plot_comp(0.4)
plt.subplot(2, 3, 4)
plot_comp(0.6)
plt.subplot(2, 3, 5)
plot_comp(0.8)
plt.subplot(2, 3, 6)
plot_comp(0.95)
plt.show()
