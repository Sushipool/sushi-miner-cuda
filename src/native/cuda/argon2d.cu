/*
MIT License

Copyright (c) 2016 Ondrej Mosnáček

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

/*
* Argon2d
* Simplified version of https://gitlab.com/omos/argon2-gpu
*/

#include "kernels.h"

__device__ uint64_t u64_build(uint32_t hi, uint32_t lo)
{
    return ((uint64_t)hi << 32) | (uint64_t)lo;
}

__device__ uint32_t u64_lo(uint64_t x)
{
    return (uint32_t)x;
}

__device__ uint32_t u64_hi(uint64_t x)
{
    return (uint32_t)(x >> 32);
}

struct block_th
{
    uint64_t a, b, c, d;
};

__device__ void move_block(struct block_th *dst, const struct block_th *src)
{
    *dst = *src;
}

__device__ void xor_block(struct block_th *dst, const struct block_th *src)
{
    dst->a ^= src->a;
    dst->b ^= src->b;
    dst->c ^= src->c;
    dst->d ^= src->d;
}

__device__ void load_block(struct block_th *dst, const struct block_g *src, uint32_t thread)
{
    dst->a = src->data[0 * THREADS_PER_LANE + thread];
    dst->b = src->data[1 * THREADS_PER_LANE + thread];
    dst->c = src->data[2 * THREADS_PER_LANE + thread];
    dst->d = src->data[3 * THREADS_PER_LANE + thread];
}

__device__ void load_block_xor(struct block_th *dst, const struct block_g *src, uint32_t thread)
{
    dst->a ^= src->data[0 * THREADS_PER_LANE + thread];
    dst->b ^= src->data[1 * THREADS_PER_LANE + thread];
    dst->c ^= src->data[2 * THREADS_PER_LANE + thread];
    dst->d ^= src->data[3 * THREADS_PER_LANE + thread];
}

__device__ void store_block(struct block_g *dst, const struct block_th *src, uint32_t thread)
{
    dst->data[0 * THREADS_PER_LANE + thread] = src->a;
    dst->data[1 * THREADS_PER_LANE + thread] = src->b;
    dst->data[2 * THREADS_PER_LANE + thread] = src->c;
    dst->data[3 * THREADS_PER_LANE + thread] = src->d;
}

__device__ uint64_t rotr64(uint64_t x, uint32_t n)
{
    return (x >> n) | (x << (64 - n));
}

__device__ uint64_t permute64(uint64_t x, uint32_t hi, uint32_t lo)
{
    uint32_t xlo = u64_lo(x);
    uint32_t xhi = u64_hi(x);
    return u64_build(__byte_perm(xlo, xhi, hi), __byte_perm(xlo, xhi, lo));
}

__device__ uint64_t f(uint64_t x, uint64_t y)
{
    uint64_t r;
    asm("{"
        ".reg .u32 xlo, ylo, mlo, mhi;"
        "cvt.u32.u64 xlo, %1;"        // xlo = u64_lo(x)
        "cvt.u32.u64 ylo, %2;"        // ylo = u64_lo(y)
        "mul.lo.u32 mlo, xlo, ylo;"   // mlo = xlo * ylo
        "mul.hi.u32 mhi, xlo, ylo;"   // mhi __umulhi(xlo, ylo)
        "mov.b64 %0, {mlo, mhi};"     // r = u64_build(mhi, mlo)
        "shl.b64 %0, %0, 1;"          // r *= 2
        "add.u64 %0, %0, %1;"         // r += x
        "add.u64 %0, %0, %2;"         // r += y
        "}"
        : "=l"(r) : "l"(x), "l"(y)
    );
    return r;
}

__device__ void g(struct block_th *block)
{
    uint64_t a, b, c, d;
    a = block->a;
    b = block->b;
    c = block->c;
    d = block->d;

    a = f(a, b);
    d = rotr64(d ^ a, 32);
    c = f(c, d);
    b = permute64(b ^ c, 0x2107, 0x6543);
    a = f(a, b);
    d = permute64(d ^ a, 0x1076, 0x5432);
    c = f(c, d);
    b = rotr64(b ^ c, 63);

    block->a = a;
    block->b = b;
    block->c = c;
    block->d = d;
}

__device__ void transpose(struct block_th *block, uint32_t thread)
{
    // thread groups, previously: thread_group = (thread & 0x0C) >> 2
    uint32_t g1 = (thread & 0x4);
    uint32_t g2 = (thread & 0x8);

    uint64_t x1 = (g2 ? (g1 ? block->c : block->d) : (g1 ? block->a : block->b));
    uint64_t x2 = (g2 ? (g1 ? block->b : block->a) : (g1 ? block->d : block->c));
    uint64_t x3 = (g2 ? (g1 ? block->a : block->b) : (g1 ? block->c : block->d));

    x1 = __shfl_xor_sync(0xFFFFFFFF, x1, 0x4);
    x2 = __shfl_xor_sync(0xFFFFFFFF, x2, 0x8);
    x3 = __shfl_xor_sync(0xFFFFFFFF, x3, 0xC);

    block->a = (g2 ? (g1 ? x3 : x2) : (g1 ? x1 : block->a));
    block->b = (g2 ? (g1 ? x2 : x3) : (g1 ? block->b : x1));
    block->c = (g2 ? (g1 ? x1 : block->c) : (g1 ? x3 : x2));
    block->d = (g2 ? (g1 ? block->d : x1) : (g1 ? x2 : x3));
}

__device__ void shift1_shuffle(struct block_th *block, uint32_t thread)
{
    uint32_t src_thr_b = (thread & 0x1c) | ((thread + 1) & 0x3);
    uint32_t src_thr_d = (thread & 0x1c) | ((thread + 3) & 0x3);

    block->b = __shfl_sync(0xFFFFFFFF, block->b, src_thr_b);
    block->c = __shfl_xor_sync(0xFFFFFFFF, block->c, 0x2);
    block->d = __shfl_sync(0xFFFFFFFF, block->d, src_thr_d);
}

__device__ void unshift1_shuffle(struct block_th *block, uint32_t thread)
{
    uint32_t src_thr_b = (thread & 0x1c) | ((thread + 3) & 0x3);
    uint32_t src_thr_d = (thread & 0x1c) | ((thread + 1) & 0x3);

    block->b = __shfl_sync(0xFFFFFFFF, block->b, src_thr_b);
    block->c = __shfl_xor_sync(0xFFFFFFFF, block->c, 0x2);
    block->d = __shfl_sync(0xFFFFFFFF, block->d, src_thr_d);
}

__device__ void shift2_shuffle(struct block_th *block, uint32_t thread)
{
    uint32_t lo = (thread & 0x1) | ((thread & 0x10) >> 3);
    uint32_t src_thr_b = (((lo + 1) & 0x2) << 3) | (thread & 0xe) | ((lo + 1) & 0x1);
    uint32_t src_thr_d = (((lo + 3) & 0x2) << 3) | (thread & 0xe) | ((lo + 3) & 0x1);

    block->b = __shfl_sync(0xFFFFFFFF, block->b, src_thr_b);
    block->c = __shfl_xor_sync(0xFFFFFFFF, block->c, 0x10);
    block->d = __shfl_sync(0xFFFFFFFF, block->d, src_thr_d);
}

__device__ void unshift2_shuffle(struct block_th *block, uint32_t thread)
{
    uint32_t lo = (thread & 0x1) | ((thread & 0x10) >> 3);
    uint32_t src_thr_b = (((lo + 3) & 0x2) << 3) | (thread & 0xe) | ((lo + 3) & 0x1);
    uint32_t src_thr_d = (((lo + 1) & 0x2) << 3) | (thread & 0xe) | ((lo + 1) & 0x1);

    block->b = __shfl_sync(0xFFFFFFFF, block->b, src_thr_b);
    block->c = __shfl_xor_sync(0xFFFFFFFF, block->c, 0x10);
    block->d = __shfl_sync(0xFFFFFFFF, block->d, src_thr_d);
}

__device__ void shuffle_block(struct block_th *block, uint32_t thread)
{
    transpose(block, thread);

    g(block);

    shift1_shuffle(block, thread);

    g(block);

    unshift1_shuffle(block, thread);
    transpose(block, thread);

    g(block);

    shift2_shuffle(block, thread);

    g(block);

    unshift2_shuffle(block, thread);
}

__device__ uint32_t compute_ref_index(struct block_th *prev, uint32_t curr_index)
{
    uint64_t v = __shfl_sync(0xFFFFFFFF, prev->a, 0);
    uint32_t ref_index = u64_lo(v);

    uint32_t ref_area_size = curr_index - 1;
    ref_index = __umulhi(ref_index, ref_index);
    ref_index = ref_area_size - 1 - __umulhi(ref_area_size, ref_index);
    return ref_index;
}

__global__ void argon2(struct block_g *memory, uint32_t cacheSize, uint32_t memoryTradeoff)
{
    extern __shared__ struct block_g cache[];
    // ref_index of the current block, -1 if current block is stored to global mem
    __shared__ uint16_t ref_indexes[MEMORY_COST];

    uint32_t job_id = blockIdx.y;
    uint32_t thread = threadIdx.x;

    // select job's memory region
    memory += (size_t)job_id * MEMORY_COST;

    struct block_th prev_prev, ref_prev, prev, tmp;
    bool is_stored = true;

    load_block(&prev_prev, memory, thread);
    load_block(&prev, memory + 1, thread);

    ((uint64_t*) ref_indexes)[0 * THREADS_PER_LANE + thread] = (uint64_t) -1;
    ((uint64_t*) ref_indexes)[1 * THREADS_PER_LANE + thread] = (uint64_t) -1;
    ((uint64_t*) ref_indexes)[2 * THREADS_PER_LANE + thread] = (uint64_t) -1;
    ((uint64_t*) ref_indexes)[3 * THREADS_PER_LANE + thread] = (uint64_t) -1;

    for (uint32_t curr_index = 2; curr_index < MEMORY_COST; curr_index++)
    {
        store_block(cache + (curr_index - 2) % cacheSize, &prev_prev, thread);
        move_block(&prev_prev, &prev);

        uint32_t ref_index = compute_ref_index(&prev, curr_index);
        uint32_t ref_ref_index = ref_indexes[ref_index];

        if (curr_index - ref_index <= cacheSize + 1)
        {
            load_block_xor(&prev, cache + ref_index % cacheSize, thread);
        }
        else if (ref_ref_index == (uint16_t) -1)
        {
            load_block_xor(&prev, memory + ref_index, thread);
        }
        else
        {
            load_block(&ref_prev, memory + ref_index - 1, thread);
            load_block_xor(&ref_prev, memory + ref_ref_index, thread);

            move_block(&tmp, &ref_prev);
            shuffle_block(&ref_prev, thread);
            xor_block(&ref_prev, &tmp);

            xor_block(&prev, &ref_prev);
        }

        move_block(&tmp, &prev);
        shuffle_block(&prev, thread);
        xor_block(&prev, &tmp);

        is_stored = !(is_stored && (curr_index >= memoryTradeoff) && (ref_ref_index == (uint16_t) -1));
        if (!is_stored)
        {
            if (thread == 0)
            {
                ref_indexes[curr_index] = ref_index;
            }
            __syncwarp();
        }
        else if (curr_index < MEMORY_COST - cacheSize - 1)
        {
            store_block(memory + curr_index, &prev, thread);
        }
    }

    store_block(memory + MEMORY_COST - 1, &prev, thread);
}
