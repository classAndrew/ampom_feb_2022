#include <iostream>
#include <cstdlib>

using namespace std;

__global__
void approx(uint64_t N, double *xv, double *yv, double *avg_dist) {
  uint64_t idx = blockDim.x*blockIdx.x + threadIdx.x;
  if (idx < N) {
    double x = xv[idx], y = yv[idx]; 
    double dist = 0;
  
    for (uint64_t i = 0; i < idx; i++) {
      double dx = xv[i]-x, dy = yv[i]-y;
      dist += sqrt(dx*dx + dy*dy)*2.0;
    }
  
    avg_dist[idx] = dist/N;
  } 
}

__global__
void choose_reject(uint64_t N, double *xv, double *yv, bool *in_circle) {
  uint64_t idx = blockDim.x*blockIdx.x + threadIdx.x;
  uint64_t stride = blockDim.x*gridDim.x;
  
  for (uint64_t i = idx; i < N; i += stride) {
    double x = xv[i]; double y = yv[i];
    in_circle[i] = (x*x + y*y) <= 1.0;
  }
}

int main(int argc, char **argv) {
  uint64_t side = argc > 1 ? atoi(argv[1]) : 128;
  uint64_t N = side*side;

  double *xv = new double[N], *yv = new double[N]; 
  double *d_xv, *d_yv;
  bool *in_circle = new bool[N], *d_in_circle;
  // cudaMallocManaged(&xv, N*sizeof(double));
  // cudaMallocManaged(&yv, N*sizeof(double));
  // cudaMallocManaged(&in_circle, N*sizeof(bool));
  cudaMalloc(&d_xv, N*sizeof(double));
  cudaMalloc(&d_yv, N*sizeof(double));
  cudaMalloc(&d_in_circle, N*sizeof(bool));
  
  for (uint64_t i = 0; i < N; i++) {
    uint64_t yc = i/side;
    uint64_t xc = i - yc*side;
    double x = -1.+2.*((double)xc/side);
    double y = -1.+2.*((double)yc/side);
    xv[i] = x;
    yv[i] = y;
  }
  
  cudaMemcpy(d_xv, xv, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_yv, yv, N*sizeof(double), cudaMemcpyHostToDevice);

  uint64_t threads = 256;
  uint64_t blocks = (N+threads-1)/threads;
  cout << "Threads: " << threads << "    " << "Blocks: " << blocks << '\n';
  choose_reject<<<blocks, threads>>>(N, d_xv, d_yv, d_in_circle);
  cudaDeviceSynchronize();

  cudaMemcpy(yv, d_yv, N*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(xv, d_xv, N*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(in_circle, d_in_circle, N*sizeof(bool), cudaMemcpyDeviceToHost);

  uint64_t pts = 0;
  for (uint64_t i = 0; i < N; i++) pts += in_circle[i];
  cout << pts << " / " << N << '\n';
 
  for (uint64_t i = 0, j = 0; i < N; i++) {
    if (in_circle[i]) {
      xv[j] = xv[i];
      yv[j] = yv[i];
      j++;
    }
  }

  cudaMemcpy(d_yv, yv, pts*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_xv, xv, pts*sizeof(double), cudaMemcpyHostToDevice);

  double *avg_dist = new double[pts], *d_avg_dist;
  // cudaMallocManaged(&avg_dist, pts*sizeof(double));
  cudaMalloc(&d_avg_dist, pts*sizeof(double)); 

  // blocks = (pts+threads-1)/threads;
  cout << "Threads: " << threads << "    " << "Blocks: " << blocks << '\n';
  approx<<<blocks, threads>>>(pts, d_xv, d_yv, d_avg_dist);
  cudaDeviceSynchronize();

  cudaMemcpy(avg_dist, d_avg_dist, pts*sizeof(double), cudaMemcpyDeviceToHost);

  double dist = 0;
  for (uint64_t i = 0; i < pts; i++) dist += avg_dist[i];
  cout << dist/pts << '\n'; 

  cudaFree(d_xv);
  cudaFree(d_yv);
  cudaFree(d_in_circle);
  cudaFree(d_avg_dist);
  delete[] xv;
  delete[] yv;
  delete[] in_circle;
  delete[] avg_dist;
}
