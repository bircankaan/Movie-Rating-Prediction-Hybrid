#include <vector>
#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>
#include <cstring>

struct Rating {
    int u, m;
    float r;
};

class RecommenderSystem {
    std::vector<std::vector<std::pair<int, float>>> items;
    std::vector<float> itemMeans;
    std::vector<int> userItemIdx;
    int maxU = 0, maxM = 0;

    static constexpr float kMinRating = 0.5f;
    static constexpr float kMaxRating = 5.0f;
    static constexpr float kDefaultRating = 3.0f;
    static constexpr float kSimilarityThreshold = 0.1f;

    static inline float clamp(const float x, const float low, const float high) noexcept {
        return std::min(std::max(x, low), high);
    }

    inline float similarity(const std::vector<std::pair<int, float>>& v1, 
                          const std::vector<std::pair<int, float>>& v2) const noexcept {
        const auto* p1 = v1.data();
        const auto* p2 = v2.data();
        const auto s1 = v1.size();
        const auto s2 = v2.size();
        size_t i1 = 0, i2 = 0;
        float xy = 0, x2 = 0, y2 = 0;
        
        const size_t min_size = std::min(s1, s2);
        while (i1 < min_size && i2 < min_size) {
            if (p1[i1].first < p2[i2].first) {
                x2 += p1[i1].second * p1[i1].second;
                ++i1;
            } else if (p2[i2].first < p1[i1].first) {
                y2 += p2[i2].second * p2[i2].second;
                ++i2;
            } else {
                const float r1 = p1[i1].second;
                const float r2 = p2[i2].second;
                xy += r1 * r2;
                x2 += r1 * r1;
                y2 += r2 * r2;
                ++i1;
                ++i2;
            }
        }

        while (i1 < s1) {
            x2 += p1[i1].second * p1[i1].second;
            ++i1;
        }
        while (i2 < s2) {
            y2 += p2[i2].second * p2[i2].second;
            ++i2;
        }

        return (x2 > 0 && y2 > 0) ? xy / std::sqrt(x2 * y2) : 0;
    }

    inline float predictItem(const int u, const int m) const noexcept {
        if (m > maxM || items[m].empty()) return kDefaultRating;

        alignas(32) float similarities[32];
        alignas(32) float differences[32];
        int nb_count = 0;
        const size_t base_idx = static_cast<size_t>(u) * (maxM + 1);

        for (int i = 0; i <= maxM && nb_count < 32; ++i) {
            const size_t idx = base_idx + i;
            if (i == m || idx >= userItemIdx.size() || userItemIdx[idx] < 0) continue;

            const float s = similarity(items[m], items[i]);
            if (s <= kSimilarityThreshold) continue;

            const auto it = std::lower_bound(items[i].begin(), items[i].end(),
                std::make_pair(u, 0.0f));
            if (it != items[i].end() && it->first == u) {
                similarities[nb_count] = s;
                differences[nb_count] = it->second - itemMeans[i];
                ++nb_count;
            }
        }

        if (nb_count == 0) return itemMeans[m] > 0 ? itemMeans[m] : kDefaultRating;

        float num = 0, den = 0;
        for (int i = 0; i < nb_count; ++i) {
            num += similarities[i] * differences[i];
            den += std::abs(similarities[i]);
        }

        return clamp(itemMeans[m] + (den > 0 ? num / den : 0), kMinRating, kMaxRating);
    }

public:
    std::vector<float> process(std::vector<Rating>& train, std::vector<Rating>& test) {
        for (const auto& r : train) {
            maxU = std::max(maxU, r.u);
            maxM = std::max(maxM, r.m);
        }

        items.resize(maxM + 1);
        itemMeans.assign(maxM + 1, 0.0f);
        userItemIdx.assign(static_cast<size_t>(maxU + 1) * (maxM + 1), -1);

        std::vector<int> itemCounts(maxM + 1);
        for (const auto& r : train) ++itemCounts[r.m];
        for (int m = 0; m <= maxM; ++m) items[m].reserve(itemCounts[m]);

        for (size_t i = 0; i < train.size(); ++i) {
            const auto& r = train[i];
            items[r.m].emplace_back(r.u, r.r);
            itemMeans[r.m] += r.r;
            const size_t idx = static_cast<size_t>(r.u) * (maxM + 1) + r.m;
            if (idx < userItemIdx.size()) userItemIdx[idx] = static_cast<int>(i);
        }


        for (int i = 0; i <= maxM; ++i) {
            const size_t size = items[i].size();
            if (size > 0) {
                itemMeans[i] *= (1.0f / static_cast<float>(size));
                std::sort(items[i].begin(), items[i].end());
            }
        }

        constexpr int factors = 10;
        std::vector<float> U((maxU + 1) * factors), V((maxM + 1) * factors);
        std::mt19937 gen(42);
        std::normal_distribution<float> dist(0, 0.1f);

        for (auto& x : U) x = dist(gen);
        for (auto& x : V) x = dist(gen);

        constexpr float reg = 0.075f;
        constexpr float lr1 = 0.008f;
        constexpr float lr2 = 0.004f;

        for (int iter = 0; iter < 70; ++iter) {
            const float lr = iter < 50 ? lr1 : lr2;
            for (const auto& r : train) {
                const size_t u_offset = r.u * factors;
                const size_t v_offset = r.m * factors;
                float pred = 0;

                for (int f = 0; f < factors; ++f) {
                    pred += U[u_offset + f] * V[v_offset + f];
                }
                const float err = r.r - pred;

                for (int f = 0; f < factors; ++f) {
                    const float oldU = U[u_offset + f];
                    U[u_offset + f] += lr * (err * V[v_offset + f] - reg * oldU);
                    V[v_offset + f] += lr * (err * oldU - reg * V[v_offset + f]);
                }
            }
        }

        std::vector<float> pred;
        pred.reserve(test.size());

        for (const auto& r : test) {
            const float p1 = predictItem(r.u, r.m);
            float p2 = kDefaultRating;

            if (r.u <= maxU && r.m <= maxM) {
                float sum = 0;
                const size_t u_offset = r.u * factors;
                const size_t v_offset = r.m * factors;

                for (int f = 0; f < factors; ++f) {
                    sum += U[u_offset + f] * V[v_offset + f];
                }
                p2 = clamp(sum, kMinRating, kMaxRating);
            }

            pred.push_back(clamp(0.46f * p1 + 0.54f * p2, kMinRating, kMaxRating));
        }

        return pred;
    }
};

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.precision(1);
    std::cout << std::fixed;

    std::vector<Rating> train, test;
    train.reserve(1000000);
    test.reserve(100000);

    char buf[256];
    bool inTrain = false;

    while (std::cin.getline(buf, 256)) {
        if (buf[0] == 't') {
            inTrain = (buf[1] == 'r');
            continue;
        }
        char* p = buf;
        Rating r = {
            static_cast<int>(std::strtol(p, &p, 10)),
            static_cast<int>(std::strtol(p, &p, 10)),
            inTrain ? std::strtof(p, &p) : 0
        };
        (inTrain ? train : test).push_back(r);
    }

    RecommenderSystem rs;
    for (float p : rs.process(train, test)) {
        std::cout << p << '\n';
    }
}
