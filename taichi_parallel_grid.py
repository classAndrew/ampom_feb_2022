# Parallelized
import time, typing, sys
import numpy as np
import taichi as ti

ti.init(arch=ti.opengl, default_fp=ti.f32, default_ip=ti.i32)

RNG = np.random # np.random.Generator(np.random.PCG64(np.random.SeedSequence().spawn(1)[0]))
side_len = int(sys.argv[1])
xx, yy = np.meshgrid(np.linspace(-1, 1, side_len), np.linspace(-1, 1, side_len))
x_np, y_np = xx.ravel().astype(np.float32), yy.ravel().astype(np.float32)
dist = x_np**2+y_np**2

x_np = x_np[dist <= 1]
y_np = y_np[dist <= 1]
x = ti.field(dtype=float, shape=x_np.shape)
y = ti.field(dtype=float, shape=y_np.shape)
result = ti.field(dtype=float, shape=3)

x.from_numpy(x_np)
y.from_numpy(y_np) 

@ti.kernel
def approx():
    dist_avg= 0.
    neighbor_count_avg = 0.

    for i in range(x_np.size):
        x_i, y_i = x[i], y[i]

        dist_sum = 0.
        neighbor_count = 0

        for j in range(i+1, x_np.size):
            x_j, y_j = x[j], y[j]
            dist = ((x_j-x_i)**2.+(y_j-y_i)**2)**.5

            # these values are multiplied by 2 because there are two ways to make the pairings (p1, p2) and (p2, p1)
            dist_sum += dist*2.
            neighbor_count += (dist < 1)*2

        dist_avg += dist_sum/x_np.size
        neighbor_count_avg += neighbor_count/x_np.size

    result[0] = x_np.size
    result[1] = dist_avg/x_np.size
    result[2] = neighbor_count_avg/x_np.size

now = time.time()
approx()
print("%7d points - Average dist: %7.7f  |  Less than 1: %7.7f  |  %9.5fs" % 
    (result[0], result[1], result[2], time.time()-now))
