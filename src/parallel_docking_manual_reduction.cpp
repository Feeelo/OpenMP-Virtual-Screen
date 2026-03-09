#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <utility>
#include <cstdlib>
#include <omp.h>

#include "utils.h"

int main(int argc, char** argv)
{
    Grid grid = read_grid("../data/grid.pts");
    std::vector<Atom> ligand_atoms = read_xyz("../data/ligand.xyz");

    // Generate poses (serial, unchanged)
    std::vector<std::vector<Atom>> poses;
    poses.reserve(1000000);
    for (int i = 0; i < 1000000; i++) {
        poses.push_back(transform_ligand(ligand_atoms, i));
    }

    // Start timing
    auto start_parallel = std::chrono::high_resolution_clock::now();

    double global_min = 10000;
    int best_pose = -1;
    std::vector<double> pose_scores(poses.size(), 0.0);


    int nthreads = omp_get_max_threads();
    std::vector<double> thread_min(nthreads, 1e100);
    std::vector<int> thread_best_pose(nthreads, -1);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        double local_min = 1e100;
        int local_best_pose = -1;

        #pragma omp for nowait
        for (int i = 0; i < static_cast<int>(poses.size()); i++) {
            double total = 0.0;
            for (const auto& atom : poses[i]) {
                total += trilinear_interp(grid, atom.x, atom.y, atom.z);
            }
            pose_scores[i] = total;
            if (total < local_min) {
                local_min = total;
                local_best_pose = i;
            }
        }

        thread_min[tid] = local_min;
        thread_best_pose[tid] = local_best_pose;
    }

    for (int t = 0; t < nthreads; ++t) {
        if (thread_min[t] < global_min) {
            global_min = thread_min[t];
            best_pose = thread_best_pose[t];
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


    // End timing
    auto end_parallel = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_parallel = end_parallel - start_parallel;

    std::cout << "Top " << top_n << " poses (lowest energy first):" << std::endl;
    for (int rank = 0; rank < top_n; ++rank) {
        std::cout << "Rank " << (rank + 1)
                  << ": energy = " << score_index[rank].first
                  << ", pose = " << score_index[rank].second << std::endl;
    }
    std::cout << "Best pose (compat) = " << best_pose
              << ", energy = " << global_min << std::endl;
    std::cout << "Time taken = " << elapsed_parallel.count() << std::endl;

    return 0;
}
