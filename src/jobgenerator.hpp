#pragma once

#include "common.hpp"

constexpr u32 ALPHABET_SIZE      = 64;
constexpr u32 MAX_PAYLOAD_LENGTH = 55;

extern u8 const alphabet[ALPHABET_SIZE];

struct LongJobGenerator {
    u8  state[MAX_PAYLOAD_LENGTH];
    u32 header_length;
    bool exhausted;

    LongJobGenerator(char const *in_header);
    bool next(u8 payload[64]);
};

struct JobDescription {
    u32 search_idxs[3];
    u32 length;
    u64 product_idx;
};

struct JobGenerator {
    u32 it_len;
    u32 it_len_max;

    u64 product_idx;
    u64 product_idx_max;

    u32 search_idxs[3];

    JobGenerator(u32 max_len);
    bool next(JobDescription &job);
};

void write_payload(JobDescription const &job, char const *header, u32 header_len, u8 *payload);

