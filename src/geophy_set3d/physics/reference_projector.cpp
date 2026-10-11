#include <cmath>
#include <algorithm>
#include <cstdint>
#include <omp.h>

// Endpoint-aligned point grid [x,y,z]. No finite-volume cell assumption.
extern "C" void project_rays(const float* volume, int n, const double* origins,
                            const double* directions, const double* near,
                            const double* far, const uint8_t* valid, int rays,
                            int rays_per_view, int samples, double m_ch4,
                            double m_air, int threads, float* output) {
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (int r = 0; r < rays; ++r) {
        if (!valid[r]) { output[r] = 0; continue; }
        const int view = r / rays_per_view;
        const double step = (far[r] - near[r]) / (samples - 1);
        double sum = 0.;
        for (int k = 0; k < samples; ++k) {
            const double t = near[r] + step * k;
            int i[3]; double f[3];
            for (int a = 0; a < 3; ++a) {
                double u = (origins[view*3+a] + t*directions[r*3+a]) * (n-1) / 0.6;
                u = std::fmax(0., std::fmin(n-1., u));
                i[a] = std::min(n-2, static_cast<int>(u));
                f[a] = u - i[a];
            }
            double c = 0.;
            for (int x = 0; x < 2; ++x)
                for (int y = 0; y < 2; ++y)
                    for (int z = 0; z < 2; ++z)
                        c += volume[((i[0]+x)*n+i[1]+y)*n+i[2]+z]
                             * (x ? f[0] : 1-f[0]) * (y ? f[1] : 1-f[1])
                             * (z ? f[2] : 1-f[2]);
            c = std::fmax(0., std::fmin(1., c));
            const double chi = (c / m_ch4) / (c / m_ch4 + (1-c) / m_air);
            sum += ((k == 0 || k == samples-1) ? 0.5 : 1.) * chi;
        }
        output[r] = static_cast<float>(sum * step * 1e6);
    }
}
