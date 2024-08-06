#pragma once

#include "common.hpp"

struct JobDescription {
    u32 search_idxs[3];
    u32 length;
    u64 product_idx;
};

// indices in ks must be in descending order and 0-based
inline bool next_combination(u32 *ks, u32 k, u32 n) {
    for (u32 i = 0; i < k; i++) {
        if (ks[i] >= n - i - 1) {
            continue;
        }

        ks[i]++;

        u32 x = ks[i];

        for (u32 j = 0; j > i; j++) {
            ks[j] = x + i - j;
        }

        return true;
    }

    return false;
}

struct JobGenerator {
    u32 it_len;
    u32 it_len_max;

    u64 product_idx;
    u64 product_idx_max;

    u32 search_idxs[3];

    JobGenerator(u32 max_len) {
        // max_len should be at least 3

        it_len     = 3;
        it_len_max = max_len;

        search_idxs[0] = 2;
        search_idxs[1] = 1;
        search_idxs[2] = 0;

        product_idx     = 0;
        product_idx_max = 1;
    }

    bool next(JobDescription &job) {
        if (it_len > it_len_max) {
            return false;
        }

        if (product_idx == product_idx_max) {
            product_idx = 0;

            bool had_next = next_combination(search_idxs, 3, it_len);

            if (!had_next) {
                search_idxs[0] = 2;
                search_idxs[1] = 1;
                search_idxs[2] = 0;

                it_len++;
                product_idx_max *= 64;

                if (it_len > it_len_max) {
                    return false;
                }
            }
        }

        job.search_idxs[0] = search_idxs[0];
        job.search_idxs[1] = search_idxs[1];
        job.search_idxs[2] = search_idxs[2];

        job.length = it_len;

        job.product_idx = product_idx;

        product_idx++;

        return true;
    }
};
