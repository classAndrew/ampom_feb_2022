# Parallelized
import time, typing, sys
import numpy as np
from numba import njit, prange

RNG = np.random # np.random.Generator(np.random.PCG64(np.random.SeedSequence().spawn(1)[0]))
CORES = 4

@njit(parallel=True, boundscheck=False)
def approx(xy) -> typing.Tuple[int, float, float]: # returns tuple of [num pts, avg dist, prob < 1]
    x, y = xy
    
    dist = x**2+y**2
    x = x[dist <= 1]
    y = y[dist <= 1]

    dist_sum = 0
    neighbor_count = 0

    for i in prange(x.size):
        x_i, y_i = x[i], y[i]
        for j in prange(i+1, x.size):
            x_j, y_j = x[j], y[j]
            dist = ((x_j-x_i)**2+(y_j-y_i)**2)**.5

            # these values are multiplied by 2 because there are two ways to make the pairings (p1, p2) and (p2, p1)
            dist_sum += dist*2
            neighbor_count += (dist < 1)*2

    return x.size, dist_sum/x.size**2, neighbor_count/x.size**2

if __name__ == "__main__":
    max_pts = int(sys.argv[1])

    if len(sys.argv) >= 3 and sys.argv[2] == "multisample":
        sys.stderr.write(f"[!] Multisampling enabled over {CORES} processes. Expect computation to take longer\n")
        now = time.time()
        # np.random isn't thread safe :(
        data = [(RNG.uniform(low=-1, high=1, size=max_pts), RNG.uniform(low=-1, high=1, size=max_pts)) for _ in range(CORES)]
        pool = mp.Pool(processes=4)
        result = np.array(pool.map(approx, data)).T
        print("%7d points - Average dist: %7.7f  |  Less than 1: %7.7f  |  %9.5fs" % 
            (np.mean(result[0]), np.mean(result[1]), np.mean(result[2]), time.time()-now))
        pool.close()
    
    else:
        sys.stderr.write(f"[!] Multisampling is disabled, add 'multisample' as an argument to enable\n")
        now = time.time()
        xx, yy = np.meshgrid(np.linspace(-1, 1, max_pts), np.linspace(-1, 1, max_pts))

