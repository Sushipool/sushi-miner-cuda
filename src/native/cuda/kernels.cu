#include "kernels.h"

__host__ uint32_t mine_nonces(struct worker_t *worker, uint32_t start_nonce, uint32_t share_compact)
{
    init_memory<<<worker->init_memory_blocks, worker->init_memory_threads>>>(worker->memory, worker->inseed, start_nonce);
    argon2<<<worker->argon2_blocks, worker->argon2_threads, ARGON2_BLOCK_SIZE>>>(worker->memory);
    get_nonce<<<worker->get_nonce_blocks, worker->get_nonce_threads>>>(worker->memory, start_nonce, share_compact, worker->nonce);

    uint32_t nonce;
    cudaMemcpy(&nonce, worker->nonce, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (nonce > 0)
    {
       cudaMemset(worker->nonce, 0, sizeof(uint32_t)); // zero nonce
    }
    return nonce;
}
