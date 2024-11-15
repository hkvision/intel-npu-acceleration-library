//
// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include "intel_npu_acceleration_library/nn_factory.h"

using namespace intel_npu_acceleration_library;
#include <iostream>
#include <chrono>

using namespace std;
using namespace std::chrono;

void read_weight_from_file_and_set_input(string model_weight_dir, string model_name,
                                         string model_layer, int idx,
                                         ov::intel_npu::level_zero::ZeroBufferTensor input_tensor)
{
    string filename = model_weight_dir + "\\" + model_name + "_" + model_layer + "_input_" + to_string(idx) + ".bin";
    int size_in_byte = input_tensor.get_byte_size();
    int8_t* argptr = (int8_t*)malloc(size_in_byte);
    FILE* fp = nullptr;
    int r = fopen_s(&fp, filename.c_str(), "rb");
    fseek(fp, 0, SEEK_END);
    size_t nSize = ftell(fp);
    // printf("read %d size bytes from %s\n", nSize, filename.c_str());
    fseek(fp, 0, SEEK_SET);
    fread(argptr, 1, nSize, fp);
    fclose(fp);
    memcpy(input_tensor.get(), argptr, size_in_byte);
}


void read_weight_from_file_and_set_to_ptr(string model_weight_dir, string model_name,
                                          string model_layer, int idx, int8_t* argptr)
{
    string filename = model_weight_dir + "\\" + model_name + "_" + model_layer + "_input_" + to_string(idx) + ".bin";
    FILE* fp = nullptr;
    int r = fopen_s(&fp, filename.c_str(), "rb");
    fseek(fp, 0, SEEK_END);
    size_t nSize = ftell(fp);
    // printf("read %d size bytes from %s\n", nSize, filename.c_str());
    fseek(fp, 0, SEEK_SET);
    fread(argptr, 1, nSize, fp);
    fclose(fp);
}

void constrcuct_prefill_mask(uint16_t* p_attention_mask, int prompt_len, int kv_len) {
    for(int i = 0 ; i < kv_len; i++) {
        for(int j = 0; j < kv_len; j++) {
            if(i < prompt_len && j <= i) {
                p_attention_mask[i * kv_len + j] = 0;
            } else {
                p_attention_mask[i * kv_len + j]  = 0xFBFF;  // binray result of -65504(torch.finfo(torch.float16).min)
            }
        }
    }
}


int main() {
    FILE* fp = nullptr;
    int r;
    size_t nSize;

    const string model_dir = "D:\\kai\\qwen2.5-full-weights-960";
    const string model_weight_dir = model_dir + "\\model_weights";


    uint32_t* input_id = new uint32_t[960];
    input_id[0] = 15469;
    input_id[1] = 102021;
    input_id[2] = 30;
    for(int i = 3; i < 960; i++){
        input_id[i] = 0;
    }
    uint16_t* attention_mask = new uint16_t[960*960];
    constrcuct_prefill_mask(attention_mask, 3, 960);
    uint64_t* position_id = new uint64_t[960];
    for (int i = 0; i < 960; i++){
        position_id[i] = i;
    }

    auto embedding_factory = std::make_shared<ModelFactory>("NPU");
    embedding_factory->create_ov_model(model_dir + "\\embedding_prefill_new.xml");
    auto hidden_buffer = embedding_factory->createRemoteOutputTensor(0);
    embedding_factory->setInputTensor(input_id, 0);
    embedding_factory->setOutputTensor(hidden_buffer, 0);
    std::cout << "embedding loaded" << std::endl;

    std::vector<half_ptr> k_results;
    std::vector<half_ptr> v_results;
    const int num_layers = 28;
    auto layer_factory = std::make_shared<ModelFactory>("NPU");
    layer_factory->create_ov_model(model_dir + "\\decoder_layer_prefill_new.xml");
    std::cout << "decoder loaded" << std::endl;
    layer_factory->setInputTensor(hidden_buffer, 0);
    layer_factory->setInputTensor(attention_mask, 1);
    layer_factory->setInputTensor(position_id, 2);
    layer_factory->setOutputTensor(hidden_buffer, 0);

    auto input_layer_norm = layer_factory->createRemoteInputTensor(3);
    auto output_layer_norm = layer_factory->createRemoteInputTensor(4);
    auto q_bias = layer_factory->createRemoteInputTensor(5);
    auto k_bias = layer_factory->createRemoteInputTensor(6);
    auto v_bias = layer_factory->createRemoteInputTensor(7);
    auto q_proj_weight = layer_factory->createRemoteInputTensor(8);
    auto q_proj_scale = layer_factory->createRemoteInputTensor(9);
    auto k_proj_weight = layer_factory->createRemoteInputTensor(10);
    auto k_proj_scale = layer_factory->createRemoteInputTensor(11);
    auto v_proj_weight = layer_factory->createRemoteInputTensor(12);
    auto v_proj_scale = layer_factory->createRemoteInputTensor(13);
    auto o_proj_weight = layer_factory->createRemoteInputTensor(14);
    auto o_proj_scale = layer_factory->createRemoteInputTensor(15);
    auto gate_proj_weight = layer_factory->createRemoteInputTensor(16);
    auto gate_proj_scale = layer_factory->createRemoteInputTensor(17);
    auto up_proj_weight = layer_factory->createRemoteInputTensor(18);
    auto up_proj_scale = layer_factory->createRemoteInputTensor(19);
    auto down_proj_weight = layer_factory->createRemoteInputTensor(20);
    auto down_proj_scale = layer_factory->createRemoteInputTensor(21);

    layer_factory->setInputTensor(input_layer_norm, 3);
    layer_factory->setInputTensor(output_layer_norm, 4);
    layer_factory->setInputTensor(q_bias, 5);
    layer_factory->setInputTensor(k_bias, 6);
    layer_factory->setInputTensor(v_bias, 7);
    layer_factory->setInputTensor(q_proj_weight, 8);
    layer_factory->setInputTensor(q_proj_scale, 9);
    layer_factory->setInputTensor(k_proj_weight, 10);
    layer_factory->setInputTensor(k_proj_scale, 11);
    layer_factory->setInputTensor(v_proj_weight, 12);
    layer_factory->setInputTensor(v_proj_scale, 13);
    layer_factory->setInputTensor(o_proj_weight, 14);
    layer_factory->setInputTensor(o_proj_scale, 15);
    layer_factory->setInputTensor(gate_proj_weight, 16);
    layer_factory->setInputTensor(gate_proj_scale, 17);
    layer_factory->setInputTensor(up_proj_weight, 18);
    layer_factory->setInputTensor(up_proj_scale, 19);
    layer_factory->setInputTensor(down_proj_weight, 20);
    layer_factory->setInputTensor(down_proj_scale, 21);

    half_ptr k_result = new uint16_t[1 * 4 * 960 * 128];
    half_ptr v_result = new uint16_t[1 * 4 * 128 * 960];
    memset(k_result, 0, 4 * 960 * 128 * sizeof(uint16_t));
    memset(v_result, 0, 4 * 960 * 128 * sizeof(uint16_t));
    layer_factory->setOutputTensor(k_result, 1);
    layer_factory->setOutputTensor(v_result, 2);

    auto lm_head_factory = std::make_shared<ModelFactory>("NPU");
    lm_head_factory->create_ov_model(model_dir + "\\lm_head_prefill_new.xml");
    std::cout << "lm head loaded" << std::endl;

    lm_head_factory->setInputTensor(hidden_buffer, 0);

    auto weight_buffer = lm_head_factory->createRemoteInputTensor(1);
    auto scale_buffer = lm_head_factory->createRemoteInputTensor(2);
    auto logits = lm_head_factory->createRemoteOutputTensor(0);
    lm_head_factory->setOutputTensor(logits, 0);
    read_weight_from_file_and_set_input(model_weight_dir, "model", "lm_head", 1, weight_buffer);
    read_weight_from_file_and_set_input(model_weight_dir, "model", "lm_head", 2, scale_buffer);
    lm_head_factory->setInputTensor(weight_buffer, 1);
    lm_head_factory->setInputTensor(scale_buffer, 2);

    const size_t N = 5;
    std::cout << "Run inference on " << N << " workloads" << std::endl;
    auto start = high_resolution_clock::now();
    for (auto k = 0; k < N; k++) {
        std::cout << k << std::endl;
        embedding_factory->run();
        for (int idx = 0; idx < num_layers; idx++) {
            std::cout << "Running layer " << idx << std::endl;
            read_weight_from_file_and_set_input(model_weight_dir, "model", to_string(idx), 3, input_layer_norm);
            read_weight_from_file_and_set_input(model_weight_dir, "model", to_string(idx), 4, output_layer_norm);
            read_weight_from_file_and_set_input(model_weight_dir, "model", to_string(idx), 5, q_bias);
            read_weight_from_file_and_set_input(model_weight_dir, "model", to_string(idx), 6, k_bias);
            read_weight_from_file_and_set_input(model_weight_dir, "model", to_string(idx), 7, v_bias);
            read_weight_from_file_and_set_input(model_weight_dir, "model", to_string(idx), 10, q_proj_weight);
            read_weight_from_file_and_set_input(model_weight_dir, "model", to_string(idx), 11, q_proj_scale);
            read_weight_from_file_and_set_input(model_weight_dir, "model", to_string(idx), 12, k_proj_weight);
            read_weight_from_file_and_set_input(model_weight_dir, "model", to_string(idx), 13, k_proj_scale);
            read_weight_from_file_and_set_input(model_weight_dir, "model", to_string(idx), 14, v_proj_weight);
            read_weight_from_file_and_set_input(model_weight_dir, "model", to_string(idx), 15, v_proj_scale);
            read_weight_from_file_and_set_input(model_weight_dir, "model", to_string(idx), 16, o_proj_weight);
            read_weight_from_file_and_set_input(model_weight_dir, "model", to_string(idx), 17, o_proj_scale);
            read_weight_from_file_and_set_input(model_weight_dir, "model", to_string(idx), 18, gate_proj_weight);
            read_weight_from_file_and_set_input(model_weight_dir, "model", to_string(idx), 19, gate_proj_scale);
            read_weight_from_file_and_set_input(model_weight_dir, "model", to_string(idx), 20, up_proj_weight);
            read_weight_from_file_and_set_input(model_weight_dir, "model", to_string(idx), 21, up_proj_scale);
            read_weight_from_file_and_set_input(model_weight_dir, "model", to_string(idx), 22, down_proj_weight);
            read_weight_from_file_and_set_input(model_weight_dir, "model", to_string(idx), 23, down_proj_scale);
            layer_factory->run();
            if (k == 0) {
                half_ptr k_result_i = new uint16_t[1 * 4 * 960 * 128];
                half_ptr v_result_i = new uint16_t[1 * 4 * 128 * 960];
                memcpy(k_result_i, k_result, 4 * 960 * 128 * 2);
                memcpy(v_result_i, v_result, 4 * 960 * 128 * 2);
                k_results.push_back(k_result_i);
                v_results.push_back(v_result_i);
            }
        }
        lm_head_factory->run();
    }

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    std::cout << "Average time used " << (double)duration.count()/N << std::endl;

    float* plogits = (float*)logits.get();
    std::cout << "logits get" << std::endl;
    std::vector<float> vlogits(plogits + 2*152064, plogits + 3*152064);
    auto result = std::max_element(vlogits.begin(), vlogits.end());
    std::cout << "New token: " << std::distance(vlogits.begin(), result) << std::endl;

    std::cout << "Inference done" << std::endl;
    system("pause");
    return 0;
}