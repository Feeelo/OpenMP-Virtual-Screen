#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <utility>

#include "utils.h"

int main(int argc, char** argv)
{
    Grid grid = read_grid("../data/grid.pts");

    std::vector<Atom> ligand_atoms = read_xyz("../data/ligand.xyz");
    // Generate poses
    std::vector<std::vector<Atom>> poses;
    for (int i = 0; i < 1000000; i++) {
        poses.push_back(transform_ligand(ligand_atoms, i));
    }

    // Start timing
    auto start_serial = std::chrono::high_resolution_clock::now();
    std::vector<std::pair<double, int>> score_index;
    score_index.reserve(poses.size());

    for (int i = 0; i < static_cast<int>(poses.size()); i++) {
        double total = 0.0;
        for (auto& atom: poses[i]) {
            total += trilinear_interp(grid, atom.x, atom.y, atom.z);
        }
        score_index.push_back({total, i});
    }

    int top_n = 5;
    if (argc > 1) {
        top_n = std::max(1, std::atoi(argv[1]));
    }
    top_n = std::min(top_n, static_cast<int>(score_index.size()));

    std::partial_sort(
        score_index.begin(),
        score_index.begin() + top_n,
        score_index.end(),
        [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
            return a.first < b.first;
        }
    );

    // End timing
    auto end_serial = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_serial = end_serial - start_serial;
    double t_serial = elapsed_serial.count();

    std::cout << "Top " << top_n << " poses (lowest energy first):" << std::endl;
    for (int rank = 0; rank < top_n; ++rank) {
        std::cout << "Rank " << (rank + 1)
                  << ": energy = " << score_index[rank].first
                  << ", pose = " << score_index[rank].second << std::endl;
    }
    std::cout << "Time taken = " << t_serial << std::endl;

    return 0;
}