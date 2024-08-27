#include "jobgenerator.hpp"

u8 const alphabet[ALPHABET_SIZE] = {  
    '/', '+', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd',
    'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
    'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
};

LongJobGenerator::LongJobGenerator(char const *header) {
    memset(state, 0, MAX_PAYLOAD_LENGTH);
    header_length = strlen(header);
    memcpy(state, header, header_length);
    exhausted = false;
}

bool LongJobGenerator::next(u64 payload[64]) {
    if (exhausted) {
        return false;
    }

    memset(payload, 0, 64);
    memcpy(job.payload, state, header_length);

    for (u32 i = header_length; i < MAX_PAYLOAD_LENGTH; i++) {
        payload[i] = alphabet[state[i]];
    }

    job.payload[61] = 0x80;
    job.payload[62] = (MAX_PAYLOAD_LENGTH * 8) / 256;
    job.payload[63] = (MAX_PAYLOAD_LENGTH * 8) % 256;

    bool carry = true;
    for (u32 i = header_length; i < MAX_PAYLOAD_LENGTH && carry; i++) {
        state[i] = (state[i] + 1) % 64;
        carry = state[header_length] == 0;
    }

    exhausted = carry;

    return true;
}

// indices in ks must be in descending order and 0-based
bool next_combination(u32 *ks, u32 k, u32 n) {
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

JobGenerator::JobGenerator(u32 max_len) {
    // max_len should be at least 3

    it_len     = 3;
    it_len_max = max_len;

    search_idxs[0] = 2;
    search_idxs[1] = 1;
    search_idxs[2] = 0;

    product_idx     = 0;
    product_idx_max = 1;
}

bool JobGenerator::next(JobDescription &job) {
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

void write_payload(JobDescription const &job, char const *header, u32 header_len, u8 payload[64]) {
    memset(payload, 0, 64);

    memcpy(payload, header, header_len);
    
    u32 total_len = header_len + job.length;
    payload[total_len] = 0x80;

    u32 bit_len = total_len * 8;
    payload[62] = bit_len / 256;
    payload[63] = bit_len % 256;

    u64 acc = job.product_idx;

    for (u32 i = 0; i < job.length; i++) {
        if (job.search_idxs[0] == i ||
            job.search_idxs[1] == i ||
            job.search_idxs[2] == i)
        {
            continue;
        }

        payload[header_len + i] = alphabet[acc % 64];

        acc /= 64;
    }
}

