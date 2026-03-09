#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <chrono>

// -----------------------------------------
// Serial vector norm
// -----------------------------------------
double norm_serial(const std::vector<double>& v) {
    double sum = 0.0;
    for (double x : v) sum += x * x;
    return std::sqrt(sum);
}

// -----------------------------------------
// Parallel vector norm - critical
// -----------------------------------------
double norm_parallel_critical(const std::vector<double>& v) {
    double sum = 0.0;

    #pragma omp parallel for
    for (size_t i = 0; i < v.size(); ++i) {
        double term = v[i] * v[i];
        #pragma omp critical
        sum += term;
    }

    return std::sqrt(sum);
}

// -----------------------------------------
// Parallel vector norm - atomic
// -----------------------------------------
double norm_parallel_atomic(const std::vector<double>& v) {
    double sum = 0.0;

    #pragma omp parallel for
    for (size_t i = 0; i < v.size(); ++i) {
        double term = v[i] * v[i];
        #pragma omp atomic
        sum += term;
    }

    return std::sqrt(sum);
}

// -----------------------------------------
// Parallel vector norm - reduction
// -----------------------------------------
double norm_parallel_reduction(const std::vector<double>& v) {
    double sum = 0.0;

    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < v.size(); ++i) {
        sum += v[i] * v[i];
    }

    return std::sqrt(sum);
}

int main() {
    // Make vector size ~50,000,000 elements (~400 MB of memory)
    const size_t N = 50000000;
    std::vector<double> v(N, 1.0);  // Fill vector with 1.0's

    // Serial timing
    auto start_serial = std::chrono::high_resolution_clock::now();
    double out_serial = norm_serial(v);
    auto end_serial = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed_serial = end_serial - start_serial;
    double t_serial = elapsed_serial.count();

    // Parallel timing: critical
    auto start_critical = std::chrono::high_resolution_clock::now();
    double out_critical = norm_parallel_critical(v);
    auto end_critical = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_critical = end_critical - start_critical;

    // Parallel timing: atomic
    auto start_atomic = std::chrono::high_resolution_clock::now();
    double out_atomic = norm_parallel_atomic(v);
    auto end_atomic = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_atomic = end_atomic - start_atomic;

    // Parallel timing: reduction
    auto start_reduction = std::chrono::high_resolution_clock::now();
    double out_reduction = norm_parallel_reduction(v);
    auto end_reduction = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_reduction = end_reduction - start_reduction;

    // Print
    std::cout << "Serial norm    = " << out_serial << "\n";
    std::cout << "Critical norm  = " << out_critical << "\n";
    std::cout << "Atomic norm    = " << out_atomic << "\n";
    std::cout << "Reduction norm = " << out_reduction << "\n\n";

    std::cout << "Serial time    = " << t_serial    << " s\n";
    std::cout << "Critical time  = " << elapsed_critical.count()  << " s\n";
    std::cout << "Atomic time    = " << elapsed_atomic.count()    << " s\n";
    std::cout << "Reduction time = " << elapsed_reduction.count() << " s\n";
    std::cout << "Critical speedup  = " << t_serial / elapsed_critical.count() << "x\n";
    std::cout << "Atomic speedup    = " << t_serial / elapsed_atomic.count() << "x\n";
    std::cout << "Reduction speedup = " << t_serial / elapsed_reduction.count() << "x\n";

    return 0;
}