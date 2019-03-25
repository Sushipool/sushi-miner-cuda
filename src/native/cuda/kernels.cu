#include "kernels.h"

__host__ uint32_t initialize_worker(struct worker_t *worker, int device, size_t memory)
{
    cudaSetDevice(device);
    cudaDeviceReset();
    cudaSetDeviceFlags(cudaDeviceScheduleAuto);
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

    uint32_t nonces_per_run = memory / (sizeof(block_g) * NIMIQ_ARGON2_COST);
    nonces_per_run = (nonces_per_run / BLAKE2B_THREADS_PER_BLOCK) * BLAKE2B_THREADS_PER_BLOCK;
    size_t mem_size = sizeof(block_g) * NIMIQ_ARGON2_COST * nonces_per_run;

    worker->device = device;
    worker->nonces_per_run = nonces_per_run;

    cudaError_t result = cudaMalloc(&worker->memory, mem_size);
    if (result != cudaSuccess)
    {
        return result;
    }

    result = cudaMalloc(&worker->inseed, sizeof(initial_seed));
    if (result != cudaSuccess)
    {
        return result;
    }

    result = cudaMalloc(&worker->nonce, sizeof(uint32_t));
    if (result != cudaSuccess)
    {
        return result;
    }

    worker->init_memory_blocks = dim3(nonces_per_run / BLAKE2B_THREADS_PER_BLOCK);
    worker->init_memory_threads = dim3(BLAKE2B_THREADS_PER_BLOCK);

    worker->argon2_blocks = dim3(1, nonces_per_run);
    worker->argon2_threads = dim3(THREADS_PER_LANE, 1);

    worker->find_nonce_blocks = dim3(nonces_per_run / BLAKE2B_THREADS_PER_BLOCK);
    worker->find_nonce_threads = dim3(BLAKE2B_THREADS_PER_BLOCK);

    return 0;
}

__host__ uint32_t set_block_header(struct worker_t *worker, nimiq_block_header *header)
{
    struct initial_seed inseed = {0};
    inseed.lanes = 1;
    inseed.hash_len = ARGON2_HASH_LENGTH;
    inseed.memory_cost = NIMIQ_ARGON2_COST;
    inseed.iterations = 1;
    inseed.version = 0x13;
    inseed.type = 0;
    inseed.header_len = sizeof(nimiq_block_header);
    memcpy(&inseed.header, header, sizeof(nimiq_block_header));
    inseed.salt_len = NIMIQ_ARGON2_SALT_LEN;
    memcpy(&inseed.salt, NIMIQ_ARGON2_SALT, NIMIQ_ARGON2_SALT_LEN);

    cudaSetDevice(worker->device);

    cudaError_t result = cudaMemcpy(worker->inseed, &inseed, sizeof(initial_seed), cudaMemcpyHostToDevice);
    if (result != cudaSuccess)
    {
        return result;
    }

    cudaMemset(worker->nonce, 0, sizeof(uint32_t)); // zero nonce

    return 0;
}

__host__ uint32_t mine_nonces(struct worker_t *worker, uint32_t start_nonce, uint32_t share_compact)
{
    init_memory<<<worker->init_memory_blocks, worker->init_memory_threads>>>(worker->memory, worker->inseed, start_nonce);
    argon2<<<worker->argon2_blocks, worker->argon2_threads, ARGON2_BLOCK_SIZE>>>(worker->memory);
    get_nonce<<<worker->find_nonce_blocks, worker->find_nonce_threads>>>(worker->memory, start_nonce, share_compact, worker->nonce);

    uint32_t nonce;
    cudaMemcpy(&nonce, worker->nonce, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (nonce > 0)
    {
       cudaMemset(worker->nonce, 0, sizeof(uint32_t)); // zero nonce
    }
    return nonce;
}

__host__ uint32_t release_worker(struct worker_t *worker)
{
    cudaFree(&worker->memory);
    cudaFree(&worker->inseed);
    cudaFree(&worker->nonce);

    return 0;
}
