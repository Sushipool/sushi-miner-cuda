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

__device__ uint64_t u64_shuffle(uint64_t v, uint32_t thread)
{
    uint32_t lo = u64_lo(v);
    uint32_t hi = u64_hi(v);
    lo = __shfl_sync(0xFFFFFFFF, lo, thread);
    hi = __shfl_sync(0xFFFFFFFF, hi, thread);
    return u64_build(hi, lo);
}

struct block_th
{
    uint64_t a, b, c, d;
};

__device__ uint64_t cmpeq_mask(uint32_t test, uint32_t ref)
{
    uint32_t x = -(uint32_t)(test == ref);
    return u64_build(x, x);
}

__device__ uint64_t block_th_get(const struct block_th *b, uint32_t idx)
{
    uint64_t res = 0;
    res ^= cmpeq_mask(idx, 0) & b->a;
    res ^= cmpeq_mask(idx, 1) & b->b;
    res ^= cmpeq_mask(idx, 2) & b->c;
    res ^= cmpeq_mask(idx, 3) & b->d;
    return res;
}

__device__ void block_th_set(struct block_th *b, uint32_t idx, uint64_t v)
{
    b->a ^= cmpeq_mask(idx, 0) & (v ^ b->a);
    b->b ^= cmpeq_mask(idx, 1) & (v ^ b->b);
    b->c ^= cmpeq_mask(idx, 2) & (v ^ b->c);
    b->d ^= cmpeq_mask(idx, 3) & (v ^ b->d);
}

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

template<class shuffle>
__device__ void apply_shuffle(struct block_th *block, uint32_t thread)
{
    for (uint32_t i = 0; i < QWORDS_PER_THREAD; i++) {
        uint32_t src_thr = shuffle::apply(thread, i);

        uint64_t v = block_th_get(block, i);
        v = u64_shuffle(v, src_thr);
        block_th_set(block, i, v);
    }
}

__device__ void transpose(struct block_th *block, uint32_t thread)
{
    uint32_t thread_group = (thread & 0x0C) >> 2;
    for (uint32_t i = 1; i < QWORDS_PER_THREAD; i++) {
        uint32_t thr = (i << 2) ^ thread;
        uint32_t idx = thread_group ^ i;

        uint64_t v = block_th_get(block, idx);
        v = u64_shuffle(v, thr);
        block_th_set(block, idx, v);
    }
}

struct shift1_shuffle {
    __device__ static uint32_t apply(uint32_t thread, uint32_t idx)
    {
        return (thread & 0x1c) | ((thread + idx) & 0x3);
    }
};

struct unshift1_shuffle {
    __device__ static uint32_t apply(uint32_t thread, uint32_t idx)
    {
        idx = (QWORDS_PER_THREAD - idx) % QWORDS_PER_THREAD;

        return (thread & 0x1c) | ((thread + idx) & 0x3);
    }
};

struct shift2_shuffle {
    __device__ static uint32_t apply(uint32_t thread, uint32_t idx)
    {
        uint32_t lo = (thread & 0x1) | ((thread & 0x10) >> 3);
        lo = (lo + idx) & 0x3;
        return ((lo & 0x2) << 3) | (thread & 0xe) | (lo & 0x1);
    }
};

struct unshift2_shuffle {
    __device__ static uint32_t apply(uint32_t thread, uint32_t idx)
    {
        idx = (QWORDS_PER_THREAD - idx) % QWORDS_PER_THREAD;

        uint32_t lo = (thread & 0x1) | ((thread & 0x10) >> 3);
        lo = (lo + idx) & 0x3;
        return ((lo & 0x2) << 3) | (thread & 0xe) | (lo & 0x1);
    }
};

__device__ void shuffle_block(struct block_th *block, uint32_t thread)
{
    transpose(block, thread);

    g(block);

    apply_shuffle<shift1_shuffle>(block, thread);

    g(block);

    apply_shuffle<unshift1_shuffle>(block, thread);
    transpose(block, thread);

    g(block);

    apply_shuffle<shift2_shuffle>(block, thread);

    g(block);

    apply_shuffle<unshift2_shuffle>(block, thread);
}

__device__ uint32_t compute_ref_index(struct block_th *prev, uint32_t curr_index)
{
    uint64_t v = u64_shuffle(prev->a, 0);
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
