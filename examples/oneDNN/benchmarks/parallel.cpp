#define _POSIX_C_SOURCE 200112L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <thread>
#include <iostream>

#include "oneapi/dnnl/dnnl.h"

#include "example_utils.h"

#define BATCH 8
#define IC 3
#define OC 96
#define CONV_IH 227
#define CONV_IW 227
#define CONV_OH 55
#define CONV_OW 55
#define CONV_STRIDE 4
#define CONV_PAD 0
#define POOL_OH 27
#define POOL_OW 27
#define POOL_STRIDE 2
#define POOL_PAD 0

#define REPEAT 30

static size_t product(dnnl_dim_t *arr, size_t size) {
    size_t prod = 1;
    for (size_t i = 0; i < size; ++i)
        prod *= arr[i];
    return prod;
}

static void init_net_data(float *data, uint32_t dim, const dnnl_dim_t *dims) {
    if (dim == 1) {
        for (dnnl_dim_t i = 0; i < dims[0]; ++i) {
            data[i] = (float)(i % 1637);
        }
    } else if (dim == 4) {
        for (dnnl_dim_t in = 0; in < dims[0]; ++in)
            for (dnnl_dim_t ic = 0; ic < dims[1]; ++ic)
                for (dnnl_dim_t ih = 0; ih < dims[2]; ++ih)
                    for (dnnl_dim_t iw = 0; iw < dims[3]; ++iw) {
                        dnnl_dim_t indx = in * dims[1] * dims[2] * dims[3]
                                + ic * dims[2] * dims[3] + ih * dims[3] + iw;
                        data[indx] = (float)(indx % 1637);
                    }
    }
}

typedef struct {
    int nargs;
    dnnl_exec_arg_t *args;
} args_t;

static void prepare_arg_node(args_t *node, int nargs) {
    node->args = (dnnl_exec_arg_t *)malloc(sizeof(dnnl_exec_arg_t) * nargs);
    node->nargs = nargs;
}
static void free_arg_node(args_t *node) {
    free(node->args);
}

static void set_arg(dnnl_exec_arg_t *arg, int arg_idx, dnnl_memory_t memory) {
    arg->arg = arg_idx;
    arg->memory = memory;
}

static void init_data_memory(uint32_t dim, const dnnl_dim_t *dims,
        dnnl_format_tag_t user_tag, dnnl_engine_t engine, float *data,
        dnnl_memory_t *memory) {
    dnnl_memory_desc_t user_md;
    CHECK(dnnl_memory_desc_create_with_tag(
            &user_md, dim, dims, dnnl_f32, user_tag));
    CHECK(dnnl_memory_create(memory, user_md, engine, DNNL_MEMORY_ALLOCATE));
    CHECK(dnnl_memory_desc_destroy(user_md));
    write_to_dnnl_memory(data, *memory);
}

dnnl_status_t prepare_reorder(dnnl_memory_t *user_memory, // in
        const_dnnl_memory_desc_t prim_memory_md, // in
        dnnl_engine_t prim_engine, // in: primitive's engine
        int dir_is_user_to_prim, // in: user -> prim or prim -> user
        dnnl_memory_t *prim_memory, // out: primitive's memory created
        dnnl_primitive_t *reorder, // out: reorder primitive created
        uint32_t *net_index, // primitive index in net (inc if reorder created)
        dnnl_primitive_t *net, args_t *net_args) { // net params
    const_dnnl_memory_desc_t user_memory_md;
    dnnl_memory_get_memory_desc(*user_memory, &user_memory_md);

    dnnl_engine_t user_mem_engine;
    dnnl_memory_get_engine(*user_memory, &user_mem_engine);

    if (!dnnl_memory_desc_equal(user_memory_md, prim_memory_md)) {
        CHECK(dnnl_memory_create(prim_memory, prim_memory_md, prim_engine,
                DNNL_MEMORY_ALLOCATE));

        dnnl_primitive_desc_t reorder_pd;
        if (dir_is_user_to_prim) {
            CHECK(dnnl_reorder_primitive_desc_create(&reorder_pd,
                    user_memory_md, user_mem_engine, prim_memory_md,
                    prim_engine, NULL));
        } else {
            CHECK(dnnl_reorder_primitive_desc_create(&reorder_pd,
                    prim_memory_md, prim_engine, user_memory_md,
                    user_mem_engine, NULL));
        }
        CHECK(dnnl_primitive_create(reorder, reorder_pd));
        CHECK(dnnl_primitive_desc_destroy(reorder_pd));

        net[*net_index] = *reorder;
        prepare_arg_node(&net_args[*net_index], 2);
        set_arg(&net_args[*net_index].args[0], DNNL_ARG_FROM,
                dir_is_user_to_prim ? *user_memory : *prim_memory);
        set_arg(&net_args[*net_index].args[1], DNNL_ARG_TO,
                dir_is_user_to_prim ? *prim_memory : *user_memory);
        (*net_index)++;
    } else {
        *prim_memory = NULL;
        *reorder = NULL;
    }

    return dnnl_success;
}

void cnn_inference(dnnl_engine_kind_t engine_kind) {
    for (int repeat = 0; repeat < REPEAT*10; repeat++) {
    dnnl_engine_t engine;
    CHECK(dnnl_engine_create(&engine, engine_kind, 0));

    // build a simple net
    uint32_t n = 0;
    dnnl_primitive_t net[10];
    args_t net_args[10];

    const int ndims = 4;
    dnnl_dims_t net_src_sizes = {BATCH, IC, CONV_IH, CONV_IW};
    dnnl_dims_t net_dst_sizes = {BATCH, OC, POOL_OH, POOL_OW};

    float *net_src
            = (float *)malloc(product(net_src_sizes, ndims) * sizeof(float));
    float *net_dst
            = (float *)malloc(product(net_dst_sizes, ndims) * sizeof(float));

    init_net_data(net_src, ndims, net_src_sizes);
    memset(net_dst, 0, product(net_dst_sizes, ndims) * sizeof(float));

    // AlexNet: conv
    // {BATCH, IC, CONV_IH, CONV_IW} (x) {OC, IC, 11, 11} ->
    // {BATCH, OC, CONV_OH, CONV_OW}
    // strides: {CONV_STRIDE, CONV_STRIDE}
    dnnl_dims_t conv_user_src_sizes;
    for (int i = 0; i < ndims; i++)
        conv_user_src_sizes[i] = net_src_sizes[i];
    dnnl_dims_t conv_user_weights_sizes = {OC, IC, 11, 11};
    dnnl_dims_t conv_bias_sizes = {OC};
    dnnl_dims_t conv_user_dst_sizes = {BATCH, OC, CONV_OH, CONV_OW};
    dnnl_dims_t conv_strides = {CONV_STRIDE, CONV_STRIDE};
    dnnl_dims_t conv_dilation = {0, 0};
    dnnl_dims_t conv_padding = {CONV_PAD, CONV_PAD};

    float *conv_src = net_src;
    float *conv_weights = (float *)malloc(
            product(conv_user_weights_sizes, ndims) * sizeof(float));
    float *conv_bias
            = (float *)malloc(product(conv_bias_sizes, 1) * sizeof(float));

    init_net_data(conv_weights, ndims, conv_user_weights_sizes);
    init_net_data(conv_bias, 1, conv_bias_sizes);

    // create memory for user data
    dnnl_memory_t conv_user_src_memory, conv_user_weights_memory,
            conv_user_bias_memory;
    init_data_memory(ndims, conv_user_src_sizes, dnnl_nchw, engine, conv_src,
            &conv_user_src_memory);
    init_data_memory(ndims, conv_user_weights_sizes, dnnl_oihw, engine,
            conv_weights, &conv_user_weights_memory);
    init_data_memory(1, conv_bias_sizes, dnnl_x, engine, conv_bias,
            &conv_user_bias_memory);

    // create data descriptors for convolution w/ no specified format

    dnnl_memory_desc_t conv_src_md, conv_weights_md, conv_bias_md, conv_dst_md;
    CHECK(dnnl_memory_desc_create_with_tag(&conv_src_md, ndims,
            conv_user_src_sizes, dnnl_f32, dnnl_format_tag_any));
    CHECK(dnnl_memory_desc_create_with_tag(&conv_weights_md, ndims,
            conv_user_weights_sizes, dnnl_f32, dnnl_format_tag_any));
    CHECK(dnnl_memory_desc_create_with_tag(
            &conv_bias_md, 1, conv_bias_sizes, dnnl_f32, dnnl_x));
    CHECK(dnnl_memory_desc_create_with_tag(&conv_dst_md, ndims,
            conv_user_dst_sizes, dnnl_f32, dnnl_format_tag_any));

    // create a convolution
    dnnl_primitive_desc_t conv_pd;
    CHECK(dnnl_convolution_forward_primitive_desc_create(&conv_pd, engine,
            dnnl_forward, dnnl_convolution_direct, conv_src_md, conv_weights_md,
            conv_bias_md, conv_dst_md, conv_strides, conv_dilation,
            conv_padding, conv_padding, NULL));

    dnnl_memory_t conv_internal_src_memory, conv_internal_weights_memory,
            conv_internal_dst_memory;

    // create memory for dst data, we don't need reorder it to user data
    const_dnnl_memory_desc_t dst_md
            = dnnl_primitive_desc_query_md(conv_pd, dnnl_query_dst_md, 0);
    CHECK(dnnl_memory_create(
            &conv_internal_dst_memory, dst_md, engine, DNNL_MEMORY_ALLOCATE));

    // create reorder primitives between user data and convolution srcs
    // if required
    dnnl_primitive_t conv_reorder_src, conv_reorder_weights;

    const_dnnl_memory_desc_t src_md
            = dnnl_primitive_desc_query_md(conv_pd, dnnl_query_src_md, 0);
    CHECK(prepare_reorder(&conv_user_src_memory, src_md, engine, 1,
            &conv_internal_src_memory, &conv_reorder_src, &n, net, net_args));

    const_dnnl_memory_desc_t weights_md
            = dnnl_primitive_desc_query_md(conv_pd, dnnl_query_weights_md, 0);
    CHECK(prepare_reorder(&conv_user_weights_memory, weights_md, engine, 1,
            &conv_internal_weights_memory, &conv_reorder_weights, &n, net,
            net_args));

    dnnl_memory_t conv_src_memory = conv_internal_src_memory
            ? conv_internal_src_memory
            : conv_user_src_memory;
    dnnl_memory_t conv_weights_memory = conv_internal_weights_memory
            ? conv_internal_weights_memory
            : conv_user_weights_memory;

    // finally create a convolution primitive
    dnnl_primitive_t conv;
    CHECK(dnnl_primitive_create(&conv, conv_pd));
    net[n] = conv;
    prepare_arg_node(&net_args[n], 4);
    set_arg(&net_args[n].args[0], DNNL_ARG_SRC, conv_src_memory);
    set_arg(&net_args[n].args[1], DNNL_ARG_WEIGHTS, conv_weights_memory);
    set_arg(&net_args[n].args[2], DNNL_ARG_BIAS, conv_user_bias_memory);
    set_arg(&net_args[n].args[3], DNNL_ARG_DST, conv_internal_dst_memory);
    n++;

    // AlexNet: relu
    // {BATCH, OC, CONV_OH, CONV_OW} -> {BATCH, OC, CONV_OH, CONV_OW}
    float negative_slope = 0.0f;

    // create relu memory descriptor on dst memory descriptor
    // from previous primitive
    const_dnnl_memory_desc_t relu_src_md
            = dnnl_primitive_desc_query_md(conv_pd, dnnl_query_dst_md, 0);
    const_dnnl_memory_desc_t relu_dst_md = relu_src_md;

    // create a relu
    dnnl_primitive_desc_t relu_pd;
    CHECK(dnnl_eltwise_forward_primitive_desc_create(&relu_pd, engine,
            dnnl_forward, dnnl_eltwise_relu, relu_src_md, relu_dst_md,
            negative_slope, 0, NULL));

    dnnl_memory_t relu_dst_memory;
    CHECK(dnnl_memory_create(
            &relu_dst_memory, relu_dst_md, engine, DNNL_MEMORY_ALLOCATE));

    // finally create a relu primitive
    dnnl_primitive_t relu;
    CHECK(dnnl_primitive_create(&relu, relu_pd));
    net[n] = relu;
    prepare_arg_node(&net_args[n], 2);
    set_arg(&net_args[n].args[0], DNNL_ARG_SRC, conv_internal_dst_memory);
    set_arg(&net_args[n].args[1], DNNL_ARG_DST, relu_dst_memory);
    n++;

    // AlexNet: lrn
    // {BATCH, OC, CONV_OH, CONV_OW} -> {BATCH, OC, CONV_OH, CONV_OW}
    // local size: 5
    // alpha: 0.0001
    // beta: 0.75
    // k: 1.0
    uint32_t local_size = 5;
    float alpha = 0.0001f;
    float beta = 0.75f;
    float k = 1.0f;

    // create lrn src memory descriptor using dst memory descriptor
    //  from previous primitive
    const_dnnl_memory_desc_t lrn_src_md = relu_dst_md;
    const_dnnl_memory_desc_t lrn_dst_md = lrn_src_md;

    // create a lrn primitive descriptor
    dnnl_primitive_desc_t lrn_pd;
    CHECK(dnnl_lrn_forward_primitive_desc_create(&lrn_pd, engine, dnnl_forward,
            dnnl_lrn_across_channels, lrn_src_md, lrn_dst_md, local_size, alpha,
            beta, k, NULL));

    // create primitives for lrn dst and workspace memory
    dnnl_memory_t lrn_dst_memory;
    CHECK(dnnl_memory_create(
            &lrn_dst_memory, lrn_dst_md, engine, DNNL_MEMORY_ALLOCATE));
    dnnl_memory_t lrn_ws_memory;
    const_dnnl_memory_desc_t lrn_ws_md
            = dnnl_primitive_desc_query_md(lrn_pd, dnnl_query_workspace_md, 0);
    CHECK(dnnl_memory_create(
            &lrn_ws_memory, lrn_ws_md, engine, DNNL_MEMORY_ALLOCATE));

    // finally create a lrn primitive
    dnnl_primitive_t lrn;
    CHECK(dnnl_primitive_create(&lrn, lrn_pd));
    net[n] = lrn;
    prepare_arg_node(&net_args[n], 3);
    set_arg(&net_args[n].args[0], DNNL_ARG_SRC, relu_dst_memory);
    set_arg(&net_args[n].args[1], DNNL_ARG_DST, lrn_dst_memory);
    set_arg(&net_args[n].args[2], DNNL_ARG_WORKSPACE, lrn_ws_memory);
    n++;

    // AlexNet: pool
    // {BATCH, OC, CONV_OH, CONV_OW} -> {BATCH, OC, POOL_OH, POOL_OW}
    // kernel: {3, 3}
    // strides: {POOL_STRIDE, POOL_STRIDE}
    // dilation: {0, 0}
    dnnl_dims_t pool_dst_sizes;
    for (int i = 0; i < ndims; i++)
        pool_dst_sizes[i] = net_dst_sizes[i];
    dnnl_dims_t pool_kernel = {3, 3};
    dnnl_dims_t pool_strides = {POOL_STRIDE, POOL_STRIDE};
    dnnl_dims_t pool_padding = {POOL_PAD, POOL_PAD};
    dnnl_dims_t pool_dilation = {0, 0};

    // create pooling memory descriptor on dst descriptor
    //  from previous primitive
    const_dnnl_memory_desc_t pool_src_md = lrn_dst_md;

    // create descriptors for dst pooling data
    dnnl_memory_desc_t pool_dst_any_md;
    CHECK(dnnl_memory_desc_create_with_tag(&pool_dst_any_md, ndims,
            pool_dst_sizes, dnnl_f32, dnnl_format_tag_any));

    // create memory for user data
    dnnl_memory_t pool_user_dst_memory;
    init_data_memory(ndims, pool_dst_sizes, dnnl_nchw, engine, net_dst,
            &pool_user_dst_memory);

    // create a pooling
    dnnl_primitive_desc_t pool_pd;
    CHECK(dnnl_pooling_forward_primitive_desc_create(&pool_pd, engine,
            dnnl_forward, dnnl_pooling_max, pool_src_md, pool_dst_any_md,
            pool_strides, pool_kernel, pool_dilation, pool_padding,
            pool_padding, NULL));

    // create memory for workspace
    dnnl_memory_t pool_ws_memory;
    const_dnnl_memory_desc_t pool_ws_md
            = dnnl_primitive_desc_query_md(pool_pd, dnnl_query_workspace_md, 0);
    CHECK(dnnl_memory_create(
            &pool_ws_memory, pool_ws_md, engine, DNNL_MEMORY_ALLOCATE));

    dnnl_memory_t pool_dst_memory;

    // create reorder primitives between user data and pooling dsts
    // if required
    dnnl_primitive_t pool_reorder_dst;
    dnnl_memory_t pool_internal_dst_memory;
    const_dnnl_memory_desc_t pool_dst_md
            = dnnl_primitive_desc_query_md(pool_pd, dnnl_query_dst_md, 0);
    n += 1; // tentative workaround: preserve space for pooling that should
            // happen before the reorder
    CHECK(prepare_reorder(&pool_user_dst_memory, pool_dst_md, engine, 0,
            &pool_internal_dst_memory, &pool_reorder_dst, &n, net, net_args));
    n -= pool_reorder_dst ? 2 : 1;

    pool_dst_memory = pool_internal_dst_memory ? pool_internal_dst_memory
                                               : pool_user_dst_memory;

    // finally create a pooling primitive
    dnnl_primitive_t pool;
    CHECK(dnnl_primitive_create(&pool, pool_pd));
    net[n] = pool;
    prepare_arg_node(&net_args[n], 3);
    set_arg(&net_args[n].args[0], DNNL_ARG_SRC, lrn_dst_memory);
    set_arg(&net_args[n].args[1], DNNL_ARG_DST, pool_dst_memory);
    set_arg(&net_args[n].args[2], DNNL_ARG_WORKSPACE, pool_ws_memory);
    n++;

    if (pool_reorder_dst) n += 1;

    dnnl_stream_t stream;
    CHECK(dnnl_stream_create(&stream, engine, dnnl_stream_default_flags));
    for (uint32_t i = 0; i < n; ++i) {
        CHECK(dnnl_primitive_execute(
                net[i], stream, net_args[i].nargs, net_args[i].args));
    }

    CHECK(dnnl_stream_wait(stream));

    // clean-up
    for (uint32_t i = 0; i < n; ++i)
        free_arg_node(&net_args[i]);

    CHECK(dnnl_primitive_desc_destroy(conv_pd));
    CHECK(dnnl_primitive_desc_destroy(relu_pd));
    CHECK(dnnl_primitive_desc_destroy(lrn_pd));
    CHECK(dnnl_primitive_desc_destroy(pool_pd));

    dnnl_stream_destroy(stream);

    free(net_src);
    free(net_dst);

    dnnl_memory_desc_destroy(conv_src_md);
    dnnl_memory_desc_destroy(conv_weights_md);
    dnnl_memory_desc_destroy(conv_bias_md);
    dnnl_memory_desc_destroy(conv_dst_md);
    dnnl_memory_desc_destroy(pool_dst_any_md);

    dnnl_memory_destroy(conv_user_src_memory);
    dnnl_memory_destroy(conv_user_weights_memory);
    dnnl_memory_destroy(conv_user_bias_memory);
    dnnl_memory_destroy(conv_internal_src_memory);
    dnnl_memory_destroy(conv_internal_weights_memory);
    dnnl_memory_destroy(conv_internal_dst_memory);
    dnnl_primitive_destroy(conv_reorder_src);
    dnnl_primitive_destroy(conv_reorder_weights);
    dnnl_primitive_destroy(conv);

    free(conv_weights);
    free(conv_bias);

    dnnl_memory_destroy(relu_dst_memory);
    dnnl_primitive_destroy(relu);

    dnnl_memory_destroy(lrn_ws_memory);
    dnnl_memory_destroy(lrn_dst_memory);
    dnnl_primitive_destroy(lrn);

    dnnl_memory_destroy(pool_user_dst_memory);
    dnnl_memory_destroy(pool_internal_dst_memory);
    dnnl_memory_destroy(pool_ws_memory);
    dnnl_primitive_destroy(pool_reorder_dst);
    dnnl_primitive_destroy(pool);

    dnnl_engine_destroy(engine);
    }
    std::cout << "cnn_inference is done" << std::endl;
}

void cnn_training() {
    for (int repeat = 0; repeat < REPEAT; repeat++) {
    dnnl_engine_t engine;
    CHECK(dnnl_engine_create(&engine, dnnl_cpu, 0)); // idx

    // build a simple net
    uint32_t n_fwd = 0, n_bwd = 0;
    dnnl_primitive_t net_fwd[10], net_bwd[10];
    args_t net_fwd_args[10], net_bwd_args[10];

    const int ndims = 4;
    dnnl_dims_t net_src_sizes = {BATCH, IC, CONV_IH, CONV_IW};
    dnnl_dims_t net_dst_sizes = {BATCH, OC, POOL_OH, POOL_OW};

    float *net_src
            = (float *)malloc(product(net_src_sizes, ndims) * sizeof(float));
    float *net_dst
            = (float *)malloc(product(net_dst_sizes, ndims) * sizeof(float));

    init_net_data(net_src, ndims, net_src_sizes);
    memset(net_dst, 0, product(net_dst_sizes, ndims) * sizeof(float));

    //----------------------------------------------------------------------
    //----------------- Forward Stream -------------------------------------
    // AlexNet: conv
    // {BATCH, IC, CONV_IH, CONV_IW} (x) {OC, IC, 11, 11} ->
    // {BATCH, OC, CONV_OH, CONV_OW}
    // strides: {CONV_STRIDE, CONV_STRIDE}
    dnnl_dims_t conv_user_src_sizes;
    for (int i = 0; i < ndims; i++)
        conv_user_src_sizes[i] = net_src_sizes[i];
    dnnl_dims_t conv_user_weights_sizes = {OC, IC, 11, 11};
    dnnl_dims_t conv_bias_sizes = {OC};
    dnnl_dims_t conv_user_dst_sizes = {BATCH, OC, CONV_OH, CONV_OW};
    dnnl_dims_t conv_strides = {CONV_STRIDE, CONV_STRIDE};
    dnnl_dims_t conv_dilation = {0, 0};
    dnnl_dims_t conv_padding = {CONV_PAD, CONV_PAD};

    float *conv_src = net_src;
    float *conv_weights = (float *)malloc(
            product(conv_user_weights_sizes, ndims) * sizeof(float));
    float *conv_bias
            = (float *)malloc(product(conv_bias_sizes, 1) * sizeof(float));

    init_net_data(conv_weights, ndims, conv_user_weights_sizes);
    init_net_data(conv_bias, 1, conv_bias_sizes);

    // create memory for user data
    dnnl_memory_t conv_user_src_memory, conv_user_weights_memory,
            conv_user_bias_memory;
    init_data_memory(ndims, conv_user_src_sizes, dnnl_nchw, engine, conv_src,
            &conv_user_src_memory);
    init_data_memory(ndims, conv_user_weights_sizes, dnnl_oihw, engine,
            conv_weights, &conv_user_weights_memory);
    init_data_memory(1, conv_bias_sizes, dnnl_x, engine, conv_bias,
            &conv_user_bias_memory);

    // create a convolution
    dnnl_primitive_desc_t conv_pd;

    {
        // create data descriptors for convolution w/ no specified format
        dnnl_memory_desc_t conv_src_md, conv_weights_md, conv_bias_md,
                conv_dst_md;
        CHECK(dnnl_memory_desc_create_with_tag(&conv_src_md, ndims,
                conv_user_src_sizes, dnnl_f32, dnnl_format_tag_any));
        CHECK(dnnl_memory_desc_create_with_tag(&conv_weights_md, ndims,
                conv_user_weights_sizes, dnnl_f32, dnnl_format_tag_any));
        CHECK(dnnl_memory_desc_create_with_tag(
                &conv_bias_md, 1, conv_bias_sizes, dnnl_f32, dnnl_x));
        CHECK(dnnl_memory_desc_create_with_tag(&conv_dst_md, ndims,
                conv_user_dst_sizes, dnnl_f32, dnnl_format_tag_any));

        CHECK(dnnl_convolution_forward_primitive_desc_create(&conv_pd, engine,
                dnnl_forward, dnnl_convolution_direct, conv_src_md,
                conv_weights_md, conv_bias_md, conv_dst_md, conv_strides,
                conv_dilation, conv_padding, conv_padding, NULL));

        CHECK(dnnl_memory_desc_destroy(conv_src_md));
        CHECK(dnnl_memory_desc_destroy(conv_weights_md));
        CHECK(dnnl_memory_desc_destroy(conv_bias_md));
        CHECK(dnnl_memory_desc_destroy(conv_dst_md));
    }

    dnnl_memory_t conv_internal_src_memory, conv_internal_weights_memory,
            conv_internal_dst_memory;

    // create memory for dst data, we don't need to reorder it to user data
    const_dnnl_memory_desc_t conv_dst_md
            = dnnl_primitive_desc_query_md(conv_pd, dnnl_query_dst_md, 0);
    CHECK(dnnl_memory_create(&conv_internal_dst_memory, conv_dst_md, engine,
            DNNL_MEMORY_ALLOCATE));

    // create reorder primitives between user data and convolution srcs
    // if required
    dnnl_primitive_t conv_reorder_src, conv_reorder_weights;

    const_dnnl_memory_desc_t conv_src_md
            = dnnl_primitive_desc_query_md(conv_pd, dnnl_query_src_md, 0);
    CHECK(prepare_reorder(&conv_user_src_memory, conv_src_md, engine, 1,
            &conv_internal_src_memory, &conv_reorder_src, &n_fwd, net_fwd,
            net_fwd_args));

    const_dnnl_memory_desc_t conv_weights_md
            = dnnl_primitive_desc_query_md(conv_pd, dnnl_query_weights_md, 0);
    CHECK(prepare_reorder(&conv_user_weights_memory, conv_weights_md, engine, 1,
            &conv_internal_weights_memory, &conv_reorder_weights, &n_fwd,
            net_fwd, net_fwd_args));

    dnnl_memory_t conv_src_memory = conv_internal_src_memory
            ? conv_internal_src_memory
            : conv_user_src_memory;
    dnnl_memory_t conv_weights_memory = conv_internal_weights_memory
            ? conv_internal_weights_memory
            : conv_user_weights_memory;

    // finally create a convolution primitive
    dnnl_primitive_t conv;
    CHECK(dnnl_primitive_create(&conv, conv_pd));
    net_fwd[n_fwd] = conv;
    prepare_arg_node(&net_fwd_args[n_fwd], 4);
    set_arg(&net_fwd_args[n_fwd].args[0], DNNL_ARG_SRC, conv_src_memory);
    set_arg(&net_fwd_args[n_fwd].args[1], DNNL_ARG_WEIGHTS,
            conv_weights_memory);
    set_arg(&net_fwd_args[n_fwd].args[2], DNNL_ARG_BIAS, conv_user_bias_memory);
    set_arg(&net_fwd_args[n_fwd].args[3], DNNL_ARG_DST,
            conv_internal_dst_memory);
    n_fwd++;

    // AlexNet: relu
    // {BATCH, OC, CONV_OH, CONV_OW} -> {BATCH, OC, CONV_OH, CONV_OW}

    float negative_slope = 0.0f;

    // keep memory format of source same as the format of convolution
    // output in order to avoid reorder
    const_dnnl_memory_desc_t relu_src_md = conv_dst_md;
    const_dnnl_memory_desc_t relu_dst_md = relu_src_md;

    // create a relu primitive descriptor
    dnnl_primitive_desc_t relu_pd;
    CHECK(dnnl_eltwise_forward_primitive_desc_create(&relu_pd, engine,
            dnnl_forward, dnnl_eltwise_relu, relu_src_md, relu_dst_md,
            negative_slope, 0, NULL));

    // create relu dst memory
    dnnl_memory_t relu_dst_memory;
    CHECK(dnnl_memory_create(
            &relu_dst_memory, relu_dst_md, engine, DNNL_MEMORY_ALLOCATE));

    // finally create a relu primitive
    dnnl_primitive_t relu;
    CHECK(dnnl_primitive_create(&relu, relu_pd));
    net_fwd[n_fwd] = relu;
    prepare_arg_node(&net_fwd_args[n_fwd], 2);
    set_arg(&net_fwd_args[n_fwd].args[0], DNNL_ARG_SRC,
            conv_internal_dst_memory);
    set_arg(&net_fwd_args[n_fwd].args[1], DNNL_ARG_DST, relu_dst_memory);
    n_fwd++;

    // AlexNet: lrn
    // {BATCH, OC, CONV_OH, CONV_OW} -> {BATCH, OC, CONV_OH, CONV_OW}
    // local size: 5
    // alpha: 0.0001
    // beta: 0.75
    // k: 1.0
    uint32_t local_size = 5;
    float alpha = 0.0001f;
    float beta = 0.75f;
    float k = 1.0f;

    // create lrn src memory descriptor using dst memory descriptor
    //  from previous primitive
    const_dnnl_memory_desc_t lrn_src_md = relu_dst_md;
    const_dnnl_memory_desc_t lrn_dst_md = lrn_src_md;

    // create a lrn primitive descriptor
    dnnl_primitive_desc_t lrn_pd;
    CHECK(dnnl_lrn_forward_primitive_desc_create(&lrn_pd, engine, dnnl_forward,
            dnnl_lrn_across_channels, lrn_src_md, lrn_dst_md, local_size, alpha,
            beta, k, NULL));

    // create primitives for lrn dst and workspace memory
    dnnl_memory_t lrn_dst_memory, lrn_ws_memory;

    CHECK(dnnl_memory_create(
            &lrn_dst_memory, lrn_dst_md, engine, DNNL_MEMORY_ALLOCATE));

    // create workspace only in training and only for forward primitive
    // query lrn_pd for workspace, this memory will be shared with forward lrn
    const_dnnl_memory_desc_t lrn_ws_md
            = dnnl_primitive_desc_query_md(lrn_pd, dnnl_query_workspace_md, 0);
    CHECK(dnnl_memory_create(
            &lrn_ws_memory, lrn_ws_md, engine, DNNL_MEMORY_ALLOCATE));

    // finally create a lrn primitive
    dnnl_primitive_t lrn;
    CHECK(dnnl_primitive_create(&lrn, lrn_pd));
    net_fwd[n_fwd] = lrn;
    prepare_arg_node(&net_fwd_args[n_fwd], 3);
    set_arg(&net_fwd_args[n_fwd].args[0], DNNL_ARG_SRC, relu_dst_memory);
    set_arg(&net_fwd_args[n_fwd].args[1], DNNL_ARG_DST, lrn_dst_memory);
    set_arg(&net_fwd_args[n_fwd].args[2], DNNL_ARG_WORKSPACE, lrn_ws_memory);
    n_fwd++;

    // AlexNet: pool
    // {BATCH, OC, CONV_OH, CONV_OW} -> {BATCH, OC, POOL_OH, POOL_OW}
    // kernel: {3, 3}
    // strides: {POOL_STRIDE, POOL_STRIDE}
    // dilation: {0, 0}
    dnnl_dims_t pool_dst_sizes;
    for (int i = 0; i < ndims; i++)
        pool_dst_sizes[i] = net_dst_sizes[i];
    dnnl_dims_t pool_kernel = {3, 3};
    dnnl_dims_t pool_strides = {POOL_STRIDE, POOL_STRIDE};
    dnnl_dims_t pool_padding = {POOL_PAD, POOL_PAD};
    dnnl_dims_t pool_dilation = {0, 0};

    // create memory for user dst data
    dnnl_memory_t pool_user_dst_memory;
    init_data_memory(4, pool_dst_sizes, dnnl_nchw, engine, net_dst,
            &pool_user_dst_memory);

    // create a pooling primitive descriptor
    dnnl_primitive_desc_t pool_pd;

    {
        // create pooling src memory descriptor using dst descriptor
        //  from previous primitive
        const_dnnl_memory_desc_t pool_src_md = lrn_dst_md;

        // create descriptors for dst pooling data
        dnnl_memory_desc_t pool_dst_md;
        CHECK(dnnl_memory_desc_create_with_tag(&pool_dst_md, 4, pool_dst_sizes,
                dnnl_f32, dnnl_format_tag_any));

        CHECK(dnnl_pooling_forward_primitive_desc_create(&pool_pd, engine,
                dnnl_forward, dnnl_pooling_max, pool_src_md, pool_dst_md,
                pool_strides, pool_kernel, pool_dilation, pool_padding,
                pool_padding, NULL));
        CHECK(dnnl_memory_desc_destroy(pool_dst_md));
    }

    // create memory for workspace
    dnnl_memory_t pool_ws_memory;
    const_dnnl_memory_desc_t pool_ws_md
            = dnnl_primitive_desc_query_md(pool_pd, dnnl_query_workspace_md, 0);
    CHECK(dnnl_memory_create(
            &pool_ws_memory, pool_ws_md, engine, DNNL_MEMORY_ALLOCATE));

    // create reorder primitives between pooling dsts and user format dst
    // if required
    dnnl_primitive_t pool_reorder_dst;
    dnnl_memory_t pool_internal_dst_memory;
    const_dnnl_memory_desc_t pool_dst_md
            = dnnl_primitive_desc_query_md(pool_pd, dnnl_query_dst_md, 0);
    n_fwd += 1; // tentative workaround: preserve space for pooling that should
            // happen before the reorder
    CHECK(prepare_reorder(&pool_user_dst_memory, pool_dst_md, engine, 0,
            &pool_internal_dst_memory, &pool_reorder_dst, &n_fwd, net_fwd,
            net_fwd_args));
    n_fwd -= pool_reorder_dst ? 2 : 1;

    dnnl_memory_t pool_dst_memory = pool_internal_dst_memory
            ? pool_internal_dst_memory
            : pool_user_dst_memory;

    // finally create a pooling primitive
    dnnl_primitive_t pool;
    CHECK(dnnl_primitive_create(&pool, pool_pd));
    net_fwd[n_fwd] = pool;
    prepare_arg_node(&net_fwd_args[n_fwd], 3);
    set_arg(&net_fwd_args[n_fwd].args[0], DNNL_ARG_SRC, lrn_dst_memory);
    set_arg(&net_fwd_args[n_fwd].args[1], DNNL_ARG_DST, pool_dst_memory);
    set_arg(&net_fwd_args[n_fwd].args[2], DNNL_ARG_WORKSPACE, pool_ws_memory);
    n_fwd++;

    if (pool_reorder_dst) n_fwd += 1;

    //-----------------------------------------------------------------------
    //----------------- Backward Stream -------------------------------------
    //-----------------------------------------------------------------------

    // ... user diff_data ...
    float *net_diff_dst
            = (float *)malloc(product(pool_dst_sizes, 4) * sizeof(float));

    init_net_data(net_diff_dst, 4, pool_dst_sizes);

    // create memory for user diff dst data
    dnnl_memory_t pool_user_diff_dst_memory;
    init_data_memory(4, pool_dst_sizes, dnnl_nchw, engine, net_diff_dst,
            &pool_user_diff_dst_memory);

    // Pooling Backward
    // pooling diff src memory descriptor
    const_dnnl_memory_desc_t pool_diff_src_md = lrn_dst_md;

    // pooling diff dst memory descriptor
    const_dnnl_memory_desc_t pool_diff_dst_md = pool_dst_md;

    // backward primitive descriptor needs to hint forward descriptor
    dnnl_primitive_desc_t pool_bwd_pd;
    CHECK(dnnl_pooling_backward_primitive_desc_create(&pool_bwd_pd, engine,
            dnnl_pooling_max, pool_diff_src_md, pool_diff_dst_md, pool_strides,
            pool_kernel, pool_dilation, pool_padding, pool_padding, pool_pd,
            NULL));

    // create reorder primitive between user diff dst and pool diff dst
    // if required
    dnnl_memory_t pool_diff_dst_memory, pool_internal_diff_dst_memory;
    dnnl_primitive_t pool_reorder_diff_dst;
    CHECK(prepare_reorder(&pool_user_diff_dst_memory, pool_diff_dst_md, engine,
            1, &pool_internal_diff_dst_memory, &pool_reorder_diff_dst, &n_bwd,
            net_bwd, net_bwd_args));

    pool_diff_dst_memory = pool_internal_diff_dst_memory
            ? pool_internal_diff_dst_memory
            : pool_user_diff_dst_memory;

    // create memory for pool diff src data
    dnnl_memory_t pool_diff_src_memory;
    CHECK(dnnl_memory_create(&pool_diff_src_memory, pool_diff_src_md, engine,
            DNNL_MEMORY_ALLOCATE));

    // finally create backward pooling primitive
    dnnl_primitive_t pool_bwd;
    CHECK(dnnl_primitive_create(&pool_bwd, pool_bwd_pd));
    net_bwd[n_bwd] = pool_bwd;
    prepare_arg_node(&net_bwd_args[n_bwd], 3);
    set_arg(&net_bwd_args[n_bwd].args[0], DNNL_ARG_DIFF_DST,
            pool_diff_dst_memory);
    set_arg(&net_bwd_args[n_bwd].args[1], DNNL_ARG_WORKSPACE, pool_ws_memory);
    set_arg(&net_bwd_args[n_bwd].args[2], DNNL_ARG_DIFF_SRC,
            pool_diff_src_memory);
    n_bwd++;

    // Backward lrn
    const_dnnl_memory_desc_t lrn_diff_dst_md = pool_diff_src_md;
    const_dnnl_memory_desc_t lrn_diff_src_md = lrn_diff_dst_md;

    // create backward lrn descriptor
    dnnl_primitive_desc_t lrn_bwd_pd;
    CHECK(dnnl_lrn_backward_primitive_desc_create(&lrn_bwd_pd, engine,
            dnnl_lrn_across_channels, lrn_diff_src_md, lrn_diff_dst_md,
            lrn_src_md, local_size, alpha, beta, k, lrn_pd, NULL));

    // create memory for lrn diff src
    dnnl_memory_t lrn_diff_src_memory;
    CHECK(dnnl_memory_create(&lrn_diff_src_memory, lrn_diff_src_md, engine,
            DNNL_MEMORY_ALLOCATE));

    // finally create backward lrn primitive
    dnnl_primitive_t lrn_bwd;
    CHECK(dnnl_primitive_create(&lrn_bwd, lrn_bwd_pd));
    net_bwd[n_bwd] = lrn_bwd;
    prepare_arg_node(&net_bwd_args[n_bwd], 4);
    set_arg(&net_bwd_args[n_bwd].args[0], DNNL_ARG_SRC, relu_dst_memory);
    set_arg(&net_bwd_args[n_bwd].args[1], DNNL_ARG_DIFF_DST,
            pool_diff_src_memory);
    set_arg(&net_bwd_args[n_bwd].args[2], DNNL_ARG_WORKSPACE, lrn_ws_memory);
    set_arg(&net_bwd_args[n_bwd].args[3], DNNL_ARG_DIFF_SRC,
            lrn_diff_src_memory);
    n_bwd++;

    // Backward relu
    const_dnnl_memory_desc_t relu_diff_src_md = lrn_diff_src_md;
    const_dnnl_memory_desc_t relu_diff_dst_md = lrn_diff_src_md;

    // create backward relu descriptor
    dnnl_primitive_desc_t relu_bwd_pd;
    CHECK(dnnl_eltwise_backward_primitive_desc_create(&relu_bwd_pd, engine,
            dnnl_eltwise_relu, relu_diff_src_md, relu_diff_dst_md, relu_src_md,
            negative_slope, 0, relu_pd, NULL));

    // create memory for relu diff src
    dnnl_memory_t relu_diff_src_memory;
    CHECK(dnnl_memory_create(&relu_diff_src_memory, relu_diff_src_md, engine,
            DNNL_MEMORY_ALLOCATE));

    // finally create backward relu primitive
    dnnl_primitive_t relu_bwd;
    CHECK(dnnl_primitive_create(&relu_bwd, relu_bwd_pd));
    net_bwd[n_bwd] = relu_bwd;
    prepare_arg_node(&net_bwd_args[n_bwd], 3);
    set_arg(&net_bwd_args[n_bwd].args[0], DNNL_ARG_SRC,
            conv_internal_dst_memory);
    set_arg(&net_bwd_args[n_bwd].args[1], DNNL_ARG_DIFF_DST,
            lrn_diff_src_memory);
    set_arg(&net_bwd_args[n_bwd].args[2], DNNL_ARG_DIFF_SRC,
            relu_diff_src_memory);
    n_bwd++;

    // Backward convolution with respect to weights
    float *conv_diff_bias_buffer
            = (float *)malloc(product(conv_bias_sizes, 1) * sizeof(float));
    float *conv_user_diff_weights_buffer = (float *)malloc(
            product(conv_user_weights_sizes, 4) * sizeof(float));

    // initialize memory for diff weights in user format
    dnnl_memory_t conv_user_diff_weights_memory;
    init_data_memory(4, conv_user_weights_sizes, dnnl_oihw, engine,
            conv_user_diff_weights_buffer, &conv_user_diff_weights_memory);

    // create backward convolution primitive descriptor
    dnnl_primitive_desc_t conv_bwd_weights_pd;

    {
        // memory descriptors should be in format `any` to allow backward
        // convolution for
        // weights to chose the format it prefers for best performance
        dnnl_memory_desc_t conv_diff_src_md, conv_diff_weights_md,
                conv_diff_bias_md, conv_diff_dst_md;
        CHECK(dnnl_memory_desc_create_with_tag(&conv_diff_src_md, 4,
                conv_user_src_sizes, dnnl_f32, dnnl_format_tag_any));
        CHECK(dnnl_memory_desc_create_with_tag(&conv_diff_weights_md, 4,
                conv_user_weights_sizes, dnnl_f32, dnnl_format_tag_any));
        CHECK(dnnl_memory_desc_create_with_tag(
                &conv_diff_bias_md, 1, conv_bias_sizes, dnnl_f32, dnnl_x));
        CHECK(dnnl_memory_desc_create_with_tag(&conv_diff_dst_md, 4,
                conv_user_dst_sizes, dnnl_f32, dnnl_format_tag_any));

        // create backward convolution descriptor
        CHECK(dnnl_convolution_backward_weights_primitive_desc_create(
                &conv_bwd_weights_pd, engine, dnnl_convolution_direct,
                conv_diff_src_md, conv_diff_weights_md, conv_diff_bias_md,
                conv_diff_dst_md, conv_strides, conv_dilation, conv_padding,
                conv_padding, conv_pd, NULL));

        CHECK(dnnl_memory_desc_destroy(conv_diff_src_md));
        CHECK(dnnl_memory_desc_destroy(conv_diff_weights_md));
        CHECK(dnnl_memory_desc_destroy(conv_diff_bias_md));
        CHECK(dnnl_memory_desc_destroy(conv_diff_dst_md));
    }

    // for best performance convolution backward might chose
    // different memory format for src and diff_dst
    // than the memory formats preferred by forward convolution
    // for src and dst respectively
    // create reorder primitives for src from forward convolution to the
    // format chosen by backward convolution
    dnnl_primitive_t conv_bwd_reorder_src;
    dnnl_memory_t conv_bwd_internal_src_memory;
    const_dnnl_memory_desc_t conv_diff_src_md = dnnl_primitive_desc_query_md(
            conv_bwd_weights_pd, dnnl_query_src_md, 0);
    CHECK(prepare_reorder(&conv_src_memory, conv_diff_src_md, engine, 1,
            &conv_bwd_internal_src_memory, &conv_bwd_reorder_src, &n_bwd,
            net_bwd, net_bwd_args));

    dnnl_memory_t conv_bwd_weights_src_memory = conv_bwd_internal_src_memory
            ? conv_bwd_internal_src_memory
            : conv_src_memory;

    // create reorder primitives for diff_dst between diff_src from relu_bwd
    // and format preferred by conv_diff_weights
    dnnl_primitive_t conv_reorder_diff_dst;
    dnnl_memory_t conv_internal_diff_dst_memory;
    const_dnnl_memory_desc_t conv_diff_dst_md = dnnl_primitive_desc_query_md(
            conv_bwd_weights_pd, dnnl_query_diff_dst_md, 0);

    CHECK(prepare_reorder(&relu_diff_src_memory, conv_diff_dst_md, engine, 1,
            &conv_internal_diff_dst_memory, &conv_reorder_diff_dst, &n_bwd,
            net_bwd, net_bwd_args));

    dnnl_memory_t conv_diff_dst_memory = conv_internal_diff_dst_memory
            ? conv_internal_diff_dst_memory
            : relu_diff_src_memory;

    // create reorder primitives for conv diff weights memory
    dnnl_primitive_t conv_reorder_diff_weights;
    dnnl_memory_t conv_internal_diff_weights_memory;
    const_dnnl_memory_desc_t conv_diff_weights_md
            = dnnl_primitive_desc_query_md(
                    conv_bwd_weights_pd, dnnl_query_diff_weights_md, 0);
    n_bwd += 1; // tentative workaround: preserve space for conv_bwd_weights
            // that should happen before the reorder

    CHECK(prepare_reorder(&conv_user_diff_weights_memory, conv_diff_weights_md,
            engine, 0, &conv_internal_diff_weights_memory,
            &conv_reorder_diff_weights, &n_bwd, net_bwd, net_bwd_args));
    n_bwd -= conv_reorder_diff_weights ? 2 : 1;

    dnnl_memory_t conv_diff_weights_memory = conv_internal_diff_weights_memory
            ? conv_internal_diff_weights_memory
            : conv_user_diff_weights_memory;

    // create memory for diff bias memory
    dnnl_memory_t conv_diff_bias_memory;
    const_dnnl_memory_desc_t conv_diff_bias_md = dnnl_primitive_desc_query_md(
            conv_bwd_weights_pd, dnnl_query_diff_weights_md, 1);
    CHECK(dnnl_memory_create(&conv_diff_bias_memory, conv_diff_bias_md, engine,
            DNNL_MEMORY_ALLOCATE));
    CHECK(dnnl_memory_set_data_handle(
            conv_diff_bias_memory, conv_diff_bias_buffer));

    // finally created backward convolution weights primitive
    dnnl_primitive_t conv_bwd_weights;
    CHECK(dnnl_primitive_create(&conv_bwd_weights, conv_bwd_weights_pd));
    net_bwd[n_bwd] = conv_bwd_weights;
    prepare_arg_node(&net_bwd_args[n_bwd], 4);
    set_arg(&net_bwd_args[n_bwd].args[0], DNNL_ARG_SRC,
            conv_bwd_weights_src_memory);
    set_arg(&net_bwd_args[n_bwd].args[1], DNNL_ARG_DIFF_DST,
            conv_diff_dst_memory);
    set_arg(&net_bwd_args[n_bwd].args[2], DNNL_ARG_DIFF_WEIGHTS,
            conv_diff_weights_memory);
    set_arg(&net_bwd_args[n_bwd].args[3], DNNL_ARG_DIFF_BIAS,
            conv_diff_bias_memory);
    n_bwd++;

    if (conv_reorder_diff_weights) n_bwd += 1;

    // output from backward stream
    void *net_diff_weights = NULL;
    void *net_diff_bias = NULL;

    int n_iter = 10; // number of iterations for training.
    dnnl_stream_t stream;
    CHECK(dnnl_stream_create(&stream, engine, dnnl_stream_default_flags));
    // Execute the net
    for (int i = 0; i < n_iter; i++) {
        for (uint32_t i = 0; i < n_fwd; ++i)
            CHECK(dnnl_primitive_execute(net_fwd[i], stream,
                    net_fwd_args[i].nargs, net_fwd_args[i].args));

        // Update net_diff_dst
        void *net_output = NULL; // output from forward stream:
        CHECK(dnnl_memory_get_data_handle(pool_user_dst_memory, &net_output));
        // ...user updates net_diff_dst using net_output...
        // some user defined func update_diff_dst(net_diff_dst, net_output)

        // Backward pass
        for (uint32_t i = 0; i < n_bwd; ++i)
            CHECK(dnnl_primitive_execute(net_bwd[i], stream,
                    net_bwd_args[i].nargs, net_bwd_args[i].args));

        // ... update weights ...
        CHECK(dnnl_memory_get_data_handle(
                conv_user_diff_weights_memory, &net_diff_weights));
        CHECK(dnnl_memory_get_data_handle(
                conv_diff_bias_memory, &net_diff_bias));
        // ...user updates weights and bias using diff weights and bias...
        // some user defined func update_weights(conv_user_weights_memory,
        // conv_bias_memory,
        //      net_diff_weights, net_diff_bias);
    }
    CHECK(dnnl_stream_wait(stream));

    dnnl_stream_destroy(stream);

    // clean up nets
    for (uint32_t i = 0; i < n_fwd; ++i)
        free_arg_node(&net_fwd_args[i]);
    for (uint32_t i = 0; i < n_bwd; ++i)
        free_arg_node(&net_bwd_args[i]);

    // Cleanup forward
    CHECK(dnnl_primitive_desc_destroy(pool_pd));
    CHECK(dnnl_primitive_desc_destroy(lrn_pd));
    CHECK(dnnl_primitive_desc_destroy(relu_pd));
    CHECK(dnnl_primitive_desc_destroy(conv_pd));

    free(net_src);
    free(net_dst);

    dnnl_memory_destroy(conv_user_src_memory);
    dnnl_memory_destroy(conv_user_weights_memory);
    dnnl_memory_destroy(conv_user_bias_memory);
    dnnl_memory_destroy(conv_internal_src_memory);
    dnnl_memory_destroy(conv_internal_weights_memory);
    dnnl_memory_destroy(conv_internal_dst_memory);
    dnnl_primitive_destroy(conv_reorder_src);
    dnnl_primitive_destroy(conv_reorder_weights);
    dnnl_primitive_destroy(conv);

    free(conv_weights);
    free(conv_bias);

    dnnl_memory_destroy(relu_dst_memory);
    dnnl_primitive_destroy(relu);

    dnnl_memory_destroy(lrn_ws_memory);
    dnnl_memory_destroy(lrn_dst_memory);
    dnnl_primitive_destroy(lrn);

    dnnl_memory_destroy(pool_user_dst_memory);
    dnnl_memory_destroy(pool_internal_dst_memory);
    dnnl_memory_destroy(pool_ws_memory);
    dnnl_primitive_destroy(pool_reorder_dst);
    dnnl_primitive_destroy(pool);

    // Cleanup backward
    CHECK(dnnl_primitive_desc_destroy(pool_bwd_pd));
    CHECK(dnnl_primitive_desc_destroy(lrn_bwd_pd));
    CHECK(dnnl_primitive_desc_destroy(relu_bwd_pd));
    CHECK(dnnl_primitive_desc_destroy(conv_bwd_weights_pd));

    dnnl_memory_destroy(pool_user_diff_dst_memory);
    dnnl_memory_destroy(pool_diff_src_memory);
    dnnl_memory_destroy(pool_internal_diff_dst_memory);
    dnnl_primitive_destroy(pool_reorder_diff_dst);
    dnnl_primitive_destroy(pool_bwd);

    free(net_diff_dst);

    dnnl_memory_destroy(lrn_diff_src_memory);
    dnnl_primitive_destroy(lrn_bwd);

    dnnl_memory_destroy(relu_diff_src_memory);
    dnnl_primitive_destroy(relu_bwd);

    dnnl_memory_destroy(conv_user_diff_weights_memory);
    dnnl_memory_destroy(conv_diff_bias_memory);
    dnnl_memory_destroy(conv_bwd_internal_src_memory);
    dnnl_primitive_destroy(conv_bwd_reorder_src);
    dnnl_memory_destroy(conv_internal_diff_dst_memory);
    dnnl_primitive_destroy(conv_reorder_diff_dst);
    dnnl_memory_destroy(conv_internal_diff_weights_memory);
    dnnl_primitive_destroy(conv_reorder_diff_weights);
    dnnl_primitive_destroy(conv_bwd_weights);

    free(conv_diff_bias_buffer);
    free(conv_user_diff_weights_buffer);

    dnnl_engine_destroy(engine);
    }
    std::cout << "cnn_training is done" << std::endl;
}

int main(int argc, char **argv) {
    std::vector<std::thread> t(2);
//     t[0] = std::thread(cnn_inference, dnnl_cpu);
    t[1] = std::thread(cnn_training);

//     t[0].join();
    t[1].join();

    return 0;
}