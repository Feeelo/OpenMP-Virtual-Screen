#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <utility>
#include <cstdlib>
#include <omp.h>

#include "utils.h"
#include "simd.h"

struct BestResult {
    double score;
    int pose_index;
};

#pragma omp declare reduction( \
    beststruct : BestResult : \
    omp_out = (omp_in.score < omp_out.score ? omp_in : omp_out) \
) initializer(omp_priv = {1e100, -1})

int main(int argc, char** argv)
{
    Grid grid = read_grid("../data/grid.pts");
    LigandSIMD ligand_atoms = read_xyz_simd("../data/ligand.xyz");

    // Generate poses (serial, unchanged)
    std::vector<LigandSIMD> poses;
    poses.reserve(1000000);
    for (int i = 0; i < 1000000; i++) {
        poses.push_back(transform_ligand_simd(ligand_atoms, i));
    }

    // Start timing for custom reduction
    auto start_parallel_custom = std::chrono::high_resolution_clock::now();

    BestResult best = {1e100, -1};
    std::vector<double> pose_scores(poses.size(), 0.0);
    #pragma omp parallel for reduction(beststruct:best) schedule(static)
    for (int i = 0; i < poses.size(); i++) {

        const auto& pose = poses[i];

        double total = 0.0;
        int n_atoms = pose.x.size();

        #pragma omp simd reduction(+:total)
        for (int a = 0; a < n_atoms; a++) {

            double x = pose.x[a];
            double y = pose.y[a];
            double z = pose.z[a];

            // inline trilinear math here
            double i_f = (x - grid.x_min) / grid.dx;
            double j_f = (y - grid.y_min) / grid.dx;
            double k_f = (z - grid.z_min) / grid.dx;

            int i0 = (int) i_f;
            int j0 = (int) j_f;
            int k0 = (int) k_f;

            i0 = std::max(0, std::min(i0, grid.n - 2));
            j0 = std::max(0, std::min(j0, grid.n - 2));
            k0 = std::max(0, std::min(k0, grid.n - 2));

            int i1 = i0 + 1;
            int j1 = j0 + 1;
            int k1 = k0 + 1;

            double xd = i_f - i0;
            double yd = j_f - j0;
            double zd = k_f - k0;

            // loads (still irregular, but now vectorized via gathers)
            double c000 = grid_value(grid, i0, j0, k0);
            double c100 = grid_value(grid, i1, j0, k0);
            double c010 = grid_value(grid, i0, j1, k0);
            double c110 = grid_value(grid, i1, j1, k0);
            double c001 = grid_value(grid, i0, j0, k1);
            double c101 = grid_value(grid, i1, j0, k1);
            double c011 = grid_value(grid, i0, j1, k1);
            double c111 = grid_value(grid, i1, j1, k1);

            double c00 = c000*(1-xd) + c100*xd;
            double c01 = c001*(1-xd) + c101*xd;
            double c10 = c010*(1-xd) + c110*xd;
            double c11 = c011*(1-xd) + c111*xd;

            double c0 = c00*(1-yd) + c10*yd;
            double c1 = c01*(1-yd) + c11*yd;

            total += c0*(1-zd) + c1*zd;
        }

        pose_scores[i] = total;

        // Update the reduction variable directly
        if (total < best.score) {
            best.score = total;
            best.pose_index  = i;
        }
    }

    int top_n = 5;
    if (argc > 1) {
        top_n = std::max(1, std::atoi(argv[1]));
    }
    top_n = std::min(top_n, static_cast<int>(pose_scores.size()));

    std::vector<std::pair<double, int>> score_index;
    score_index.reserve(pose_scores.size());
    for (int i = 0; i < static_cast<int>(pose_scores.size()); ++i) {
        score_index.push_back({pose_scores[i], i});
    }

    std::partial_sort(
        score_index.begin(),
        score_index.begin() + top_n,
        score_index.end(),
        [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
            return a.first < b.first;
        }
    );
    auto end_parallel_custom = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_parallel_custom = end_parallel_custom - start_parallel_custom;
    // End timing for custom reduction

    std::cout << "Top " << top_n << " poses (lowest energy first):" << std::endl;
    for (int rank = 0; rank < top_n; ++rank) {
        std::cout << "Rank " << (rank + 1)
                  << ": energy = " << score_index[rank].first
                  << ", pose = " << score_index[rank].second << std::endl;
    }
    std::cout << "Best pose (compat) = " << best.pose_index
              << ", energy = " << best.score << std::endl;
    std::cout << "Time taken = " << elapsed_parallel_custom.count() << std::endl;
    return 0;
}