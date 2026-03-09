#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <utility>
#include <cstdlib>
#include <omp.h>

#include "utils.h"

struct BestResult {
    double score;
    int pose_index;
};

#pragma omp declare reduction(best_result_reduction : BestResult : \
    omp_out = (omp_in.score < omp_out.score ? omp_in : omp_out)) \
    initializer(omp_priv = BestResult{1e100, -1})

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
    auto start_parallel_custom = std::chrono::high_resolution_clock::now();

    BestResult best = {1e100, -1};
    std::vector<double> pose_scores(poses.size(), 0.0);


    #pragma omp parallel for reduction(best_result_reduction:best)
    for (int i = 0; i < static_cast<int>(poses.size()); i++) {
        double total = 0.0;
        for (const auto& atom : poses[i]) {
            total += trilinear_interp(grid, atom.x, atom.y, atom.z);
        }
        pose_scores[i] = total;
        if (total < best.score) {
            best.score = total;
            best.pose_index = i;
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
    auto end_parallel_custom = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_parallel_custom = end_parallel_custom - start_parallel_custom;

    std::cout << "Top " << top_n << " poses:" << std::endl;
    for (int rank = 0; rank < top_n; ++rank) {
        std::cout << "Rank " << (rank + 1)
                  << ": Interpolated value = " << score_index[rank].first
                  << ", pose = " << score_index[rank].second << std::endl;
    }
    std::cout << "Best pose (compat) = " << best.pose_index
              << ", Interpolated value = " << best.score << std::endl;
    std::cout << "Time taken = " << elapsed_parallel_custom.count() << std::endl;
    return 0;
}
