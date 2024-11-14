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


int main() {
    FILE* fp = nullptr;
    int r;
    size_t nSize;

    const string model_dir = "D:\\kai\\remote-tensor\\qwen-dumps";
    const string model_weight_dir = model_dir + "\\model_weights";


    uint32_t* input_id = new uint32_t[1];
    input_id[0] = 15469;
    uint64_t* attention_mask = new uint64_t[1024];
    int64_t minVal = std::numeric_limits<int64_t>::min();
    for(int i = 0; i < 1024; i++){
        attention_mask[i] = minVal;
    }
    uint64_t* position_id = new uint64_t[1];
    position_id[0] = 0;
    attention_mask[1023] = 0;

    auto embedding_factory = std::make_shared<ModelFactory>("NPU");
    embedding_factory->create_ov_model(model_dir + "\\embedding_new.xml");
    auto hidden_buffer = embedding_factory->createRemoteOutputTensor(0);
    embedding_factory->setInputTensor(input_id, 0);
    embedding_factory->setOutputTensor(hidden_buffer, 0);

    std::vector<std::shared_ptr<ModelFactory>> decoder_layers;
    std::vector<half_ptr> k_caches;
    std::vector<half_ptr> v_caches;
    std::vector<half_ptr> k_results;
    std::vector<half_ptr> v_results;
    const int num_layers = 28;
    for (int idx = 0; idx < num_layers; idx++) {
        std::cout << "Loading layer " << idx << std::endl;
        auto layer_factory = std::make_shared<ModelFactory>("NPU");
        layer_factory->create_ov_model(model_dir + "\\decoder_layer_"+std::to_string(idx)+"_new.xml");
        layer_factory->setInputTensor(hidden_buffer, 0);
        layer_factory->setInputTensor(attention_mask, 1);
        layer_factory->setInputTensor(position_id, 2);

        auto input_layer_norm = layer_factory->createRemoteInputTensor(3);
        auto output_layer_norm = layer_factory->createRemoteInputTensor(4);
        auto q_bias = layer_factory->createRemoteInputTensor(5);
        auto k_bias = layer_factory->createRemoteInputTensor(6);
        auto v_bias = layer_factory->createRemoteInputTensor(7);
        auto q_proj_weight = layer_factory->createRemoteInputTensor(10);
        auto q_proj_scale = layer_factory->createRemoteInputTensor(11);
        auto k_proj_weight = layer_factory->createRemoteInputTensor(12);
        auto k_proj_scale = layer_factory->createRemoteInputTensor(13);
        auto v_proj_weight = layer_factory->createRemoteInputTensor(14);
        auto v_proj_scale = layer_factory->createRemoteInputTensor(15);
        auto o_proj_weight = layer_factory->createRemoteInputTensor(16);
        auto o_proj_scale = layer_factory->createRemoteInputTensor(17);
        auto gate_proj_weight = layer_factory->createRemoteInputTensor(18);
        auto gate_proj_scale = layer_factory->createRemoteInputTensor(19);
        auto up_proj_weight = layer_factory->createRemoteInputTensor(20);
        auto up_proj_scale = layer_factory->createRemoteInputTensor(21);
        auto down_proj_weight = layer_factory->createRemoteInputTensor(22);
        auto down_proj_scale = layer_factory->createRemoteInputTensor(23);

        layer_factory->setInputTensor(input_layer_norm, 3);
        layer_factory->setInputTensor(output_layer_norm, 4);
        layer_factory->setInputTensor(q_bias, 5);
        layer_factory->setInputTensor(k_bias, 6);
        layer_factory->setInputTensor(v_bias, 7);
        layer_factory->setInputTensor(q_proj_weight, 10);
        layer_factory->setInputTensor(q_proj_scale, 11);
        layer_factory->setInputTensor(k_proj_weight, 12);
        layer_factory->setInputTensor(k_proj_scale, 13);
        layer_factory->setInputTensor(v_proj_weight, 14);
        layer_factory->setInputTensor(v_proj_scale, 15);
        layer_factory->setInputTensor(o_proj_weight, 16);
        layer_factory->setInputTensor(o_proj_scale, 17);
        layer_factory->setInputTensor(gate_proj_weight, 18);
        layer_factory->setInputTensor(gate_proj_scale, 19);
        layer_factory->setInputTensor(up_proj_weight, 20);
        layer_factory->setInputTensor(up_proj_scale, 21);
        layer_factory->setInputTensor(down_proj_weight, 22);
        layer_factory->setInputTensor(down_proj_scale, 23);

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

        half_ptr k_cache = new uint16_t[1 * 4 * 1023 * 128];
        half_ptr v_cache = new uint16_t[1 * 4 * 128 * 1023];
        memset(k_cache, 0, 4 * 1023 * 128 * sizeof(uint16_t));
        memset(v_cache, 0, 4 * 1023 * 128 * sizeof(uint16_t));
        layer_factory->setInputTensor(k_cache, 8);
        layer_factory->setInputTensor(v_cache, 9);
        k_caches.push_back(k_cache);
        v_caches.push_back(v_cache);

        layer_factory->setOutputTensor(hidden_buffer, 0);
        half_ptr k_result = new uint16_t[1 * 4 * 128];
        half_ptr v_result = new uint16_t[1 * 4 * 128];
        memset(k_result, 0, 4 * 128 * sizeof(uint16_t));
        memset(v_result, 0, 4 * 128 * sizeof(uint16_t));
        layer_factory->setOutputTensor(k_result, 1);
        layer_factory->setOutputTensor(v_result, 2);
        k_results.push_back(k_result);
        v_results.push_back(v_result);
        decoder_layers.push_back(layer_factory);
    }

    auto lm_head_factory = std::make_shared<ModelFactory>("NPU");
    lm_head_factory->create_ov_model(model_dir + "\\lm_head_new.xml");

    lm_head_factory->setInputTensor(hidden_buffer, 0);

    fp = nullptr;
    r = fopen_s(&fp, "D:/kai/remote-tensor/qwen-dumps/model_weights/model_lm_head_input_1.bin", "rb");
    fseek(fp, 0, SEEK_END);
    nSize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    uint8_t* lm_head_weight_buffer = new uint8_t[28 * 152064 * 128 / 2];
    fread(lm_head_weight_buffer, 1, nSize, fp);
    auto lm_head_weight = Tensor(ov::element::Type_t::i4, ov::Shape({28, 152064, 128}), lm_head_weight_buffer);
    lm_head_factory->setInputTensor(lm_head_weight.get_tensor(), 1);

    fp = nullptr;
    r = fopen_s(&fp, "D:/kai/remote-tensor/qwen-dumps/model_weights/model_lm_head_input_2.bin", "rb");
    fseek(fp, 0, SEEK_END);
    nSize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    half_ptr lm_head_scale_buffer = new uint16_t[28 * 152064];
    fread(lm_head_scale_buffer, 2, nSize/2, fp);
    auto lm_head_scale = Tensor(ov::element::Type_t::f16, ov::Shape({28, 1, 152064}), lm_head_scale_buffer);
    lm_head_factory->setInputTensor(lm_head_scale.get_tensor(), 2);

    auto logits = lm_head_factory->createRemoteOutputTensor(0);
    lm_head_factory->setOutputTensor(logits, 0);

    // prefill
    std::vector<int> prompts = {15469, 102021, 30};
    input_id[0] = prompts[0];
    const size_t N = 50;
    std::cout << "Run inference on " << N << " workloads" << std::endl;
    auto start = high_resolution_clock::now();
    for (auto k = 0; k < N; k++) {
        embedding_factory->run();
        for (int idx = 0; idx < num_layers; idx++) {
            decoder_layers[idx]->run();
        }
        lm_head_factory->run();
        for (int idx = 0; idx < num_layers; idx++) {
            half_ptr k_cache = k_caches[idx];
            half_ptr new_k = k_results[idx];
            for (int i = 0; i < 4; i++)
            {
                uint16_t* pdst = k_cache + i * 1023 * 128 + position_id[0] * 128;
                uint16_t* psrc = new_k + i * 128;
                memcpy(pdst, psrc, 128 * 2);
            }
            half_ptr v_cache = v_caches[idx];
            half_ptr new_v = v_results[idx];
            for (int i = 0; i < 4 * 128; i++)
            {
                v_cache[i * 1023 + position_id[0]] = new_v[i];
            }
        }
        // std::cout << "k: " << unsigned(k_results[0][0]) << " " << unsigned(k_results[0][1]) << std::endl;
        // std::cout << "k: " << unsigned(k_results[0][510]) << " " << unsigned(k_results[0][511]) << std::endl;
        // std::cout << "v: " << unsigned(v_results[0][0]) << " " << unsigned(v_results[0][1]) << std::endl;
        // std::cout << "v: " << unsigned(v_results[0][510]) << " " << unsigned(v_results[0][511]) << std::endl;
        // system("pause");
        if (k < prompts.size() - 1) {  // prefill
            input_id[0] = prompts[k + 1];
        }
        else {
            float* plogits = (float*)logits.get();
            // std::cout << plogits[0] << " " << plogits[1] << std::endl;
            std::vector<float> vlogits(plogits, plogits + 152064);
            auto result = std::max_element(vlogits.begin(), vlogits.end());
            // Should be updated in place
            input_id[0] = std::distance(vlogits.begin(), result);
            std::cout << "New token: " << input_id[0] << std::endl;
        }
        position_id[0] = k + 1;
        attention_mask[position_id[0] - 1] = 0;
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    std::cout << "Average time used " << (double)duration.count()/N << std::endl;
    std::cout << "Inference done" << std::endl;
    system("pause");
    return 0;
}