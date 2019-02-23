#ifndef KERNELS_H
#define KERNELS_H

#include <stdint.h>

#define ONE_GB (1 << 30)
#define ONE_MB (1 << 20)

#define ARGON2_BLOCK_SIZE 1024
#define ARGON2_QWORDS_IN_BLOCK (ARGON2_BLOCK_SIZE / 8)

#define THREADS_PER_LANE 32
#define QWORDS_PER_THREAD (ARGON2_QWORDS_IN_BLOCK / 32)

#define BLAKE2B_HASH_LENGTH 64
#define BLAKE2B_BLOCK_SIZE 128
#define BLAKE2B_QWORDS_IN_BLOCK (BLAKE2B_BLOCK_SIZE / 8)
#define BLAKE2B_THREADS_PER_BLOCK 128

#define ARGON2_HASH_LENGTH 32
#define ARGON2_INITIAL_SEED_SIZE 197
#define ARGON2_PREHASH_SEED_SIZE 76

#define NIMIQ_ARGON2_SALT "nimiqrocks!"
#define NIMIQ_ARGON2_SALT_LEN 11
#define NIMIQ_ARGON2_COST 512

struct __attribute__((packed)) nimiq_block_header
{
    // Big endian
    uint16_t version;
    uint8_t prev_hash[32];
    uint8_t interlink_hash[32];
    uint8_t body_hash[32];
    uint8_t accounts_hash[32];
    uint32_t nbits;
    uint32_t height;
    uint32_t timestamp;
    uint32_t nonce;
};

struct __attribute__((packed)) initial_seed
{
    uint32_t lanes;
    uint32_t hash_len;
    uint32_t memory_cost;
    uint32_t iterations;
    uint32_t version;
    uint32_t type;
    uint32_t header_len;
    nimiq_block_header header;
    uint32_t salt_len;
    uint8_t salt[NIMIQ_ARGON2_SALT_LEN];
    uint32_t secret_len;
    uint32_t extra_len;
    uint8_t padding[59];
};

struct block_g
{
    uint64_t data[ARGON2_QWORDS_IN_BLOCK];
};

struct worker_t
{
    int device;
    uint32_t nonces_per_run;
    block_g *memory;
    uint64_t *inseed;
    uint32_t *nonce;
    dim3 init_memory_blocks;
    dim3 init_memory_threads;
    dim3 argon2_blocks;
    dim3 argon2_threads;
    dim3 find_nonce_blocks;
    dim3 find_nonce_threads;
};

__global__ void init_memory(struct block_g *memory, uint64_t *inseed, uint32_t memory_cost, uint32_t start_nonce);
__global__ void argon2d(struct block_g *memory, uint32_t memory_cost);
__global__ void get_nonce(struct block_g *memory, uint32_t memory_cost, uint32_t start_nonce, uint32_t share_compact, uint32_t *nonce);

__host__ uint32_t initialize_worker(struct worker_t *worker, int device, size_t memory);
__host__ uint32_t set_block_header(struct worker_t *worker, nimiq_block_header *header);
__host__ uint32_t mine_nonces(struct worker_t *worker, uint32_t start_nonce, uint32_t share_compact);
__host__ uint32_t release_worker(struct worker_t *worker);

#endif
