#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "example_utils.h"
#include "oneapi/dnnl/dnnl.h"

#include "VLC/runtime.h"
#include "VLC/loader.h"

std::chrono::_V2::system_clock::time_point vlc_init_start;
std::chrono::_V2::system_clock::time_point vlc_init_end;

void matmul_example(int vlc_id, dnnl_engine_kind_t engine_kind) {
    VLC::Context vlc(vlc_id, gettid());
    vlc.avaliable_cpu("0-23");
    VLC::register_vlc(&vlc);
    VLC::Loader loader("/usr/local/lib/libdnnl.so.3", vlc_id, true);
    vlc_init_end = std::chrono::high_resolution_clock::now();
    std::cout << "PERF: VLC init finished in " << std::chrono::duration_cast<std::chrono::milliseconds>(vlc_init_end - vlc_init_start).count() << "ms" << std::endl;

    dnnl_engine_t engine;
    CHECK(dnnl_engine_create(&engine, engine_kind, 0));

    dnnl_stream_t stream;
    CHECK(dnnl_stream_create(&stream, engine, dnnl_stream_default_flags));

    const int64_t M = 512, K = 512, N = 512;

    int64_t src_dims[] = {M, K};
    int64_t weights_dims[] = {K, N};
    int64_t dst_dims[] = {M, N};

    float *src_data = (float*)malloc(M * K * sizeof(float));
    float *weights_data = (float*)malloc(K * N * sizeof(float));
    float *dst_data = (float*)malloc(M * N * sizeof(float));

    // Initialize src, weights
    for (int i = 0; i < M * K; i++) {
        src_data[i] = cosf(i / 10.f);
    }

    int64_t nnz = 0;
    for (int i = 0; i < K * N; i++) {
        if (rand() % 10 == 0) {
            weights_data[i] = sinf(i * 2.f);
            nnz++;
        } else {
            weights_data[i] = 0.0f;
        }
    }

    // Create memory descriptors
    dnnl_memory_desc_t src_md, weights_md, dst_md;
    CHECK(dnnl_memory_desc_create_with_tag(&src_md, 2, src_dims, dnnl_f32, dnnl_ab));
    CHECK(dnnl_memory_desc_create_with_tag(&weights_md, 2, weights_dims, dnnl_f32, dnnl_ab));
    CHECK(dnnl_memory_desc_create_with_tag(&dst_md, 2, dst_dims, dnnl_f32, dnnl_ab));

    // Create memory objects
    dnnl_memory_t src_mem, weights_mem, dst_mem;
    CHECK(dnnl_memory_create(&src_mem, src_md, engine, DNNL_MEMORY_ALLOCATE));
    CHECK(dnnl_memory_create(&weights_mem, weights_md, engine, DNNL_MEMORY_ALLOCATE));
    CHECK(dnnl_memory_create(&dst_mem, dst_md, engine, DNNL_MEMORY_ALLOCATE));

    // Write data to memory objects
    void *src_handle, *weights_handle, *dst_handle;
    CHECK(dnnl_memory_get_data_handle(src_mem, &src_handle));
    CHECK(dnnl_memory_get_data_handle(weights_mem, &weights_handle));
    CHECK(dnnl_memory_get_data_handle(dst_mem, &dst_handle));

    memcpy(src_handle, src_data, M * K * sizeof(float));
    memcpy(weights_handle, weights_data, K * N * sizeof(float));

    // Create matmul primitive descriptor
    dnnl_primitive_desc_t matmul_pd;
    CHECK(dnnl_matmul_primitive_desc_create(&matmul_pd, engine, src_md, weights_md, NULL, dst_md, NULL));

    // Create matmul primitive
    dnnl_primitive_t matmul_prim;
    CHECK(dnnl_primitive_create(&matmul_prim, matmul_pd));

    // Execute primitive
    dnnl_exec_arg_t args[] = {
        {DNNL_ARG_SRC, src_mem},
        {DNNL_ARG_WEIGHTS, weights_mem},
        {DNNL_ARG_DST, dst_mem}
    };

    CHECK(dnnl_primitive_execute(matmul_prim, stream, sizeof(args) / sizeof(args[0]), args));

    // Wait for execution to finish
    CHECK(dnnl_stream_wait(stream));

    // Read result
    memcpy(dst_data, dst_handle, M * N * sizeof(float));

    // Clean up
    CHECK(dnnl_primitive_desc_destroy(matmul_pd));
    CHECK(dnnl_primitive_destroy(matmul_prim));
    CHECK(dnnl_memory_destroy(src_mem));
    CHECK(dnnl_memory_destroy(weights_mem));
    CHECK(dnnl_memory_destroy(dst_mem));
    CHECK(dnnl_stream_destroy(stream));
    CHECK(dnnl_engine_destroy(engine));

    free(src_data);
    free(weights_data);
    free(dst_data);
}

int main(int argc, char **argv) {
    vlc_init_start = std::chrono::high_resolution_clock::now();
    VLC::Runtime vlc; // initialize VLC environment
    vlc.initialize();
    std::thread t;
    t = std::thread(matmul_example, 1, dnnl_cpu);
    t.join();
    printf("Example passed on CPU.\n");
    return 0;
}