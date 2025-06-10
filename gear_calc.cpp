#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <mutex>
#include <omp.h>

using namespace std;


double target_ratio;
int stages, min_teeth, max_teeth;
double max_error;

vector<int> best_a_factors, best_b_factors;
double best_error = numeric_limits<double>::max();
int best_teeth = numeric_limits<int>::max();
mutex best_mutex;

unordered_map<int, vector<int>> factor_cache;
mutex cache_mutex;

bool factor(int n, int depth, vector<int>& result, int stages) {
    if (depth == stages) return n == 1;

    vector<int> best_factors;
    int best_sum = numeric_limits<int>::max();

    for (int i = min_teeth; i <= max_teeth; ++i) {
        if (n % i == 0) {
            vector<int> temp = {i};
            if (factor(n / i, depth + 1, temp, stages)) {
                int sum = accumulate(temp.begin(), temp.end(), 0);
                if (sum < best_sum) {
                    best_sum = sum;
                    best_factors = temp;
                }
            }
        }
    }

    if (!best_factors.empty()) {
        result.insert(result.end(), best_factors.begin(), best_factors.end());
        return true;
    }

    return false;
}



void search() {
    int min_a = pow(min_teeth, stages);
    int max_a = pow(max_teeth, stages);

    #pragma omp parallel for schedule(dynamic)
    for (int a = min_a; a <= max_a; ++a) {
        int b = round(a * target_ratio);
        double ratio = static_cast<double>(b) / a;
        double error = abs(ratio - target_ratio);
        if (error > max_error) continue;


        vector<int> a_factors, b_factors;
        if (factor(a, 0, a_factors, stages) && factor(b, 0, b_factors, stages)) {
            int total_teeth = accumulate(a_factors.begin(), a_factors.end(), 0) +
                                accumulate(b_factors.begin(), b_factors.end(), 0);

            lock_guard<mutex> lock(best_mutex);
            if (error < best_error || (abs(error - best_error) < 1e-12 && total_teeth < best_teeth)) {
                best_error = error;
                best_teeth = total_teeth;
                best_a_factors = a_factors;
                best_b_factors = b_factors;
            }
        }
    }
}


int main() {
    cout << "Enter desired gear ratio (e.g., 75): ";
    cin >> target_ratio;
    cout << "Enter number of compound stages (e.g., 2): ";
    cin >> stages;
    cout << "Enter minimum number of teeth per gear (e.g., 12): ";
    cin >> min_teeth;
    cout << "Enter maximum number of teeth per gear (e.g., 100): ";
    cin >> max_teeth;
    cout << "Enter maximum allowed error (e.g., 0.0001): ";
    cin >> max_error;


    auto start = chrono::high_resolution_clock::now();
    search();
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    cout << "\nSearch completed in " << elapsed.count() << " seconds.\n";

    if (!best_a_factors.empty()) {
        cout << "\nBest approximation:\nCompound Result\n";
        for (int i = 0; i < stages; ++i) {
            cout << best_a_factors[i] << ":" << best_b_factors[i];
            if (i != stages - 1) cout << ",";
        }
        cout << "\nTeeth: " << best_teeth << endl;

        double achieved = 1.0;
        for (int i = 0; i < stages; ++i)
            achieved *= static_cast<double>(best_b_factors[i]) / best_a_factors[i];

        double error = achieved - target_ratio;

        cout << fixed << setprecision(15);
        cout << "Error: " << showpos << error << endl;
        cout << noshowpos << target_ratio << " <- target" << endl;
        cout << achieved << " <- gears" << endl;

        int matching_digits = 0;
        string target_str = to_string(target_ratio);
        string achieved_str = to_string(achieved);
        while (matching_digits < target_str.size() &&
               matching_digits < achieved_str.size() &&
               target_str[matching_digits] == achieved_str[matching_digits]) {
            ++matching_digits;
        }
        cout << "Accurate to " << (matching_digits - 2) << " decimal digits.\n";

        double deviation = abs(error / target_ratio);
        cout << "\nDeviation: " << fixed << setprecision(6) << deviation * 100 << "%" << endl;
        if (deviation > 0) {
            cout << "1 in " << static_cast<long long>(1.0 / deviation) << endl;
        }
    } else {
        cout << "\nNo valid gear combination found within the specified constraints.\n";
    }

    return 0;
}
