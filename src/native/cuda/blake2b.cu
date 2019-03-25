#include "kernels.h"


#define IV0 0x6a09e667f3bcc908UL
#define IV1 0xbb67ae8584caa73bUL
#define IV2 0x3c6ef372fe94f82bUL
#define IV3 0xa54ff53a5f1d36f1UL
#define IV4 0x510e527fade682d1UL
#define IV5 0x9b05688c2b3e6c1fUL
#define IV6 0x1f83d9abfb41bd6bUL
#define IV7 0x5be0cd19137e2179UL

struct __attribute__((packed)) prehash_seed
{
    uint32_t hashlen;
    uint64_t initial_hash[8];
    uint32_t block;
    uint32_t lane;
    uint32_t padding[13];
};

__device__ __forceinline__ uint64_t rotr64(uint64_t x, uint32_t n)
{
    return (x >> n) | (x << (64 - n));
}

__device__ void blake2b_init(uint64_t *h, uint32_t hashlen)
{
    h[0] = IV0 ^ (0x01010000 | hashlen);
    h[1] = IV1;
    h[2] = IV2;
    h[3] = IV3;
    h[4] = IV4;
    h[5] = IV5;
    h[6] = IV6;
    h[7] = IV7;
}

#define G(a, b, c, d, x, y)             \
    do                                  \
    {                                   \
        v[a] = v[a] + v[b] + m[x];      \
        v[d] = rotr64(v[d] ^ v[a], 32); \
        v[c] = v[c] + v[d];             \
        v[b] = rotr64(v[b] ^ v[c], 24); \
        v[a] = v[a] + v[b] + m[y];      \
        v[d] = rotr64(v[d] ^ v[a], 16); \
        v[c] = v[c] + v[d];             \
        v[b] = rotr64(v[b] ^ v[c], 63); \
    } while (0)

__device__ void blake2b_compress(uint64_t *h, uint64_t *m, uint32_t bytes_compressed, bool last_block)
{
    uint64_t v[BLAKE2B_QWORDS_IN_BLOCK];

    v[0] = h[0];
    v[1] = h[1];
    v[2] = h[2];
    v[3] = h[3];
    v[4] = h[4];
    v[5] = h[5];
    v[6] = h[6];
    v[7] = h[7];
    v[8] = IV0;
    v[9] = IV1;
    v[10] = IV2;
    v[11] = IV3;
    v[12] = IV4 ^ bytes_compressed;
    v[13] = IV5; // it's OK if below 2^32 bytes
    v[14] = last_block ? ~IV6 : IV6;
    v[15] = IV7;

    // Round 0
    G(0, 4, 8, 12, 0, 1);
    G(1, 5, 9, 13, 2, 3);
    G(2, 6, 10, 14, 4, 5);
    G(3, 7, 11, 15, 6, 7);
    G(0, 5, 10, 15, 8, 9);
    G(1, 6, 11, 12, 10, 11);
    G(2, 7, 8, 13, 12, 13);
    G(3, 4, 9, 14, 14, 15);
    // Round 1
    G(0, 4, 8, 12, 14, 10);
    G(1, 5, 9, 13, 4, 8);
    G(2, 6, 10, 14, 9, 15);
    G(3, 7, 11, 15, 13, 6);
    G(0, 5, 10, 15, 1, 12);
    G(1, 6, 11, 12, 0, 2);
    G(2, 7, 8, 13, 11, 7);
    G(3, 4, 9, 14, 5, 3);
    // Round 2
    G(0, 4, 8, 12, 11, 8);
    G(1, 5, 9, 13, 12, 0);
    G(2, 6, 10, 14, 5, 2);
    G(3, 7, 11, 15, 15, 13);
    G(0, 5, 10, 15, 10, 14);
    G(1, 6, 11, 12, 3, 6);
    G(2, 7, 8, 13, 7, 1);
    G(3, 4, 9, 14, 9, 4); 
    // Round 3
    G(0, 4, 8, 12, 7, 9);
    G(1, 5, 9, 13, 3, 1);
    G(2, 6, 10, 14, 13, 12);
    G(3, 7, 11, 15, 11, 14);
    G(0, 5, 10, 15, 2, 6);
    G(1, 6, 11, 12, 5, 10);
    G(2, 7, 8, 13, 4, 0);
    G(3, 4, 9, 14, 15, 8);
    // Round 4
    G(0, 4, 8, 12, 9, 0);
    G(1, 5, 9, 13, 5, 7);
    G(2, 6, 10, 14, 2, 4);
    G(3, 7, 11, 15, 10, 15);
    G(0, 5, 10, 15, 14, 1);
    G(1, 6, 11, 12, 11, 12);
    G(2, 7, 8, 13, 6, 8);
    G(3, 4, 9, 14, 3, 13); 
    // Round 5
    G(0, 4, 8, 12, 2, 12);
    G(1, 5, 9, 13, 6, 10);
    G(2, 6, 10, 14, 0, 11);
    G(3, 7, 11, 15, 8, 3);
    G(0, 5, 10, 15, 4, 13);
    G(1, 6, 11, 12, 7, 5);
    G(2, 7, 8, 13, 15, 14);
    G(3, 4, 9, 14, 1, 9);
    // Round 6
    G(0, 4, 8, 12, 12, 5);
    G(1, 5, 9, 13, 1, 15);
    G(2, 6, 10, 14, 14, 13);
    G(3, 7, 11, 15, 4, 10);
    G(0, 5, 10, 15, 0, 7);
    G(1, 6, 11, 12, 6, 3);
    G(2, 7, 8, 13, 9, 2);
    G(3, 4, 9, 14, 8, 11);
    // Round 7
    G(0, 4, 8, 12, 13, 11);
    G(1, 5, 9, 13, 7, 14);
    G(2, 6, 10, 14, 12, 1);
    G(3, 7, 11, 15, 3, 9);
    G(0, 5, 10, 15, 5, 0);
    G(1, 6, 11, 12, 15, 4);
    G(2, 7, 8, 13, 8, 6);
    G(3, 4, 9, 14, 2, 10);
    // Round 8
    G(0, 4, 8, 12, 6, 15);
    G(1, 5, 9, 13, 14, 9);
    G(2, 6, 10, 14, 11, 3);
    G(3, 7, 11, 15, 0, 8);
    G(0, 5, 10, 15, 12, 2);
    G(1, 6, 11, 12, 13, 7);
    G(2, 7, 8, 13, 1, 4);
    G(3, 4, 9, 14, 10, 5);
    // Round 9
    G(0, 4, 8, 12, 10, 2);
    G(1, 5, 9, 13, 8, 4);
    G(2, 6, 10, 14, 7, 6);
    G(3, 7, 11, 15, 1, 5);
    G(0, 5, 10, 15, 15, 11);
    G(1, 6, 11, 12, 9, 14);
    G(2, 7, 8, 13, 3, 12);
    G(3, 4, 9, 14, 13, 0);
    // Round 10
    G(0, 4, 8, 12, 0, 1);
    G(1, 5, 9, 13, 2, 3);
    G(2, 6, 10, 14, 4, 5);
    G(3, 7, 11, 15, 6, 7);
    G(0, 5, 10, 15, 8, 9);
    G(1, 6, 11, 12, 10, 11);
    G(2, 7, 8, 13, 12, 13);
    G(3, 4, 9, 14, 14, 15);
    // Round 11
    G(0, 4, 8, 12, 14, 10);
    G(1, 5, 9, 13, 4, 8);
    G(2, 6, 10, 14, 9, 15);
    G(3, 7, 11, 15, 13, 6);
    G(0, 5, 10, 15, 1, 12);
    G(1, 6, 11, 12, 0, 2);
    G(2, 7, 8, 13, 11, 7);
    G(3, 4, 9, 14, 5, 3);

    h[0] = h[0] ^ v[0] ^ v[8];
    h[1] = h[1] ^ v[1] ^ v[9];
    h[2] = h[2] ^ v[2] ^ v[10];
    h[3] = h[3] ^ v[3] ^ v[11];
    h[4] = h[4] ^ v[4] ^ v[12];
    h[5] = h[5] ^ v[5] ^ v[13];
    h[6] = h[6] ^ v[6] ^ v[14];
    h[7] = h[7] ^ v[7] ^ v[15];
}

__device__ __forceinline__ uint32_t swap32(uint32_t value)
{
    return ((value & 0xFF000000) >> 24)
        | ((value & 0x00FF0000) >> 8)
        | ((value & 0x0000FF00) << 8)
        | ((value & 0x000000FF) << 24);
}

__device__ __forceinline__ void set_nonce(uint64_t *inseed, uint32_t nonce)
{
    // bytes 170-173
    uint64_t n = swap32(nonce);
    inseed[21] = (inseed[21] & 0xFFFF00000000FFFFUL) | (n << 16);
}

__device__ void initial_hash(uint64_t *inseed, uint32_t nonce, uint64_t *hash)
{
    uint64_t is[32];
    memcpy(is, inseed, sizeof(is));
    set_nonce(is, nonce);

    blake2b_init(hash, BLAKE2B_HASH_LENGTH);
    blake2b_compress(hash, &is[0], BLAKE2B_BLOCK_SIZE, false);
    blake2b_compress(hash, &is[BLAKE2B_QWORDS_IN_BLOCK], ARGON2_INITIAL_SEED_SIZE, true);
}

__device__ void fill_block(struct prehash_seed *phseed, struct block_g *memory)
{
    uint64_t h[8];
    uint64_t buffer[BLAKE2B_QWORDS_IN_BLOCK] = {0};
    uint64_t *dst = memory->data;

    // V1
    blake2b_init(h, BLAKE2B_HASH_LENGTH);
    blake2b_compress(h, (uint64_t *)phseed, ARGON2_PREHASH_SEED_SIZE, true);

    *(dst++) = h[0];
    *(dst++) = h[1];
    *(dst++) = h[2];
    *(dst++) = h[3];

    // V2-Vr
    for (int r = 2; r < 2 * ARGON2_BLOCK_SIZE / BLAKE2B_HASH_LENGTH; r++)
    {
        buffer[0] = h[0];
        buffer[1] = h[1];
        buffer[2] = h[2];
        buffer[3] = h[3];
        buffer[4] = h[4];
        buffer[5] = h[5];
        buffer[6] = h[6];
        buffer[7] = h[7];

        blake2b_init(h, BLAKE2B_HASH_LENGTH);
        blake2b_compress(h, buffer, BLAKE2B_HASH_LENGTH, true);

        *(dst++) = h[0];
        *(dst++) = h[1];
        *(dst++) = h[2];
        *(dst++) = h[3];
    }

    *(dst++) = h[4];
    *(dst++) = h[5];
    *(dst++) = h[6];
    *(dst++) = h[7];
}

__device__ void fill_first_blocks(uint64_t *inseed, struct block_g *memory, uint32_t nonce)
{
    struct prehash_seed phs = {ARGON2_BLOCK_SIZE};

    initial_hash(inseed, nonce, phs.initial_hash);

    phs.block = 0;
    fill_block(&phs, memory);

    phs.block = 1;
    fill_block(&phs, memory + 1);
}

__device__ void compact_to_target(uint32_t share_compact, uint8_t *target)
{
    uint32_t offset = (31 - (share_compact >> 24)); // offset in bytes
    uint32_t value = share_compact & 0x00FFFFFF;

#pragma unroll
    for (int i = 0; i < ARGON2_HASH_LENGTH; i++)
    {
        target[i] = 0;
    }
    target[++offset] = (uint8_t)(value >> 16);
    target[++offset] = (uint8_t)(value >> 8);
    target[++offset] = (uint8_t)(value);
}

__device__ bool is_proof_of_work(uint8_t *hash, uint8_t *target)
{
#pragma unroll
    for (int i = 0; i < ARGON2_HASH_LENGTH; i++)
    {
        if (hash[i] < target[i])
            return true;
        if (hash[i] > target[i])
            return false;
    }
    return true;
}

__device__ void hash_last_block(struct block_g *memory, uint64_t *hash)
{
    uint64_t h[8];
    uint64_t buffer[BLAKE2B_QWORDS_IN_BLOCK];
    uint32_t hi, lo;
    uint32_t bytes_compressed = 0;
    uint32_t bytes_remaining = ARGON2_BLOCK_SIZE;
    uint32_t *src = (uint32_t *)memory->data;

    blake2b_init(h, ARGON2_HASH_LENGTH);

    hi = *(src++);
    buffer[0] = ARGON2_HASH_LENGTH | ((uint64_t)hi << 32);

#pragma unroll
    for (int i = 1; i < BLAKE2B_QWORDS_IN_BLOCK; i++)
    {
        lo = *(src++);
        hi = *(src++);
        buffer[i] = lo | ((uint64_t)hi << 32);
    }

    bytes_compressed += BLAKE2B_BLOCK_SIZE;
    bytes_remaining -= (BLAKE2B_BLOCK_SIZE - sizeof(uint32_t));
    blake2b_compress(h, buffer, bytes_compressed, false);

    while (bytes_remaining > BLAKE2B_BLOCK_SIZE)
    {
#pragma unroll
        for (int i = 0; i < BLAKE2B_QWORDS_IN_BLOCK; i++)
        {
            lo = *(src++);
            hi = *(src++);
            buffer[i] = lo | ((uint64_t)hi << 32);
        }
        bytes_compressed += BLAKE2B_BLOCK_SIZE;
        bytes_remaining -= BLAKE2B_BLOCK_SIZE;
        blake2b_compress(h, buffer, bytes_compressed, false);
    }

    buffer[0] = *src;
#pragma unroll
    for (int i = 1; i < BLAKE2B_QWORDS_IN_BLOCK; i++)
    {
        buffer[i] = 0;
    }
    bytes_compressed += bytes_remaining;
    blake2b_compress(h, buffer, bytes_compressed, true);

    hash[0] = h[0];
    hash[1] = h[1];
    hash[2] = h[2];
    hash[3] = h[3];
}

__global__ void init_memory(struct block_g *memory, uint64_t *inseed, uint32_t start_nonce)
{
    uint32_t thread = blockIdx.x * blockDim.x + threadIdx.x;
    memory += (size_t)thread * MEMORY_COST;
    fill_first_blocks(inseed, memory, start_nonce + thread);
}

__global__ void get_nonce(struct block_g *memory, uint32_t start_nonce, uint32_t share_compact, uint32_t *nonce)
{
    uint32_t thread = blockIdx.x * blockDim.x + threadIdx.x;
    memory += (size_t)(thread + 1) * MEMORY_COST - 1;

    uint8_t hash[ARGON2_HASH_LENGTH];
    uint8_t target[ARGON2_HASH_LENGTH];

    compact_to_target(share_compact, target);
    hash_last_block(memory, (uint64_t *)hash);

    if (is_proof_of_work(hash, target))
    {
        atomicCAS(nonce, 0, start_nonce + thread);
    }
}
