//
// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include "intel_npu_acceleration_library/nn_factory.h"

using namespace intel_npu_acceleration_library;
#include <iostream>
#include <chrono>
using namespace std::chrono;

int main() {
    const int num_layers = 28;
    FILE* fp = nullptr;
    int r;
    size_t nSize;
    // Using remote tensors
    // std::vector<ov::intel_npu::level_zero::ZeroBufferTensor> hidden_buffers;
    // Using ordinary buffers
    // std::vector<half_ptr> hidden_buffers;

    uint32_t* input_id = new uint32_t[1];
    input_id[0] = 100;
    // std::cout << input_id[0] << std::endl;
    uint64_t* attention_mask = new uint64_t[1024];
    int64_t minVal = std::numeric_limits<int64_t>::min();
    for(int i = 0; i < 1024; i++){
        attention_mask[i] = minVal;
    }
    uint64_t* position_id = new uint64_t[1];
    position_id[0] = 0;

    // Embedding
    auto embedding_factory = std::make_shared<ModelFactory>("NPU");
    embedding_factory->create_ov_model("C:/Users/SAS/kai/remote-tensor/qwen-dumps/embedding_new.xml");

    // half_ptr embed_buffer = new uint16_t[3584];
    auto hidden_buffer = embedding_factory->createRemoteOutputTensor(0);
    // hidden_buffers.push_back(embed_buffer);
    embedding_factory->setInputTensor(input_id, 0);
    embedding_factory->setOutputTensor(hidden_buffer, 0);
    system("pause");

    // Decoder
    std::vector<std::shared_ptr<ModelFactory>> decoder_layers;
    std::vector<half_ptr> k_caches;
    std::vector<half_ptr> v_caches;
    std::vector<half_ptr> k_results;
    std::vector<half_ptr> v_results;
    std::vector<Tensor> weight_buffers;
    for (int idx = 0; idx < num_layers; idx++) {
        std::cout << "Loading layer " << idx << std::endl;
        auto layer_factory = std::make_shared<ModelFactory>("NPU");
        layer_factory->create_ov_model("C:/Users/SAS/kai/remote-tensor/qwen-dumps/decoder_layer_"+std::to_string(idx)+"_new.xml");

        // Current version is layernorm as const for qwen
        r = fopen_s(&fp, ("C:/Users/SAS/kai/remote-tensor/qwen-dumps/model_weights/model_"+std::to_string(idx)+"_input_3.bin").c_str(), "rb");
        fseek(fp, 0, SEEK_END);
        nSize = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        half_ptr q_bias_buffer = new uint16_t[3584];
        fread(q_bias_buffer, 1, nSize/2, fp);
        std::cout << "q_bias: " << unsigned(q_bias_buffer[0]) << " " << unsigned(q_bias_buffer[1]) << std::endl;
        auto q_bias = Tensor(ov::element::Type_t::f16, ov::Shape({1, 3584}), q_bias_buffer);
        weight_buffers.push_back(q_bias);
        // half_ptr q_bias_ptr = (half_ptr)q_bias.data();
        // half_buffers.push_back(q_bias_buffer);
        // std::cout << "q_bias: " << unsigned(q_bias_ptr[0]) << " " << unsigned(q_bias_ptr[0]) << std::endl;
        // system("pause");

        fp = nullptr;
        r = fopen_s(&fp, ("C:/Users/SAS/kai/remote-tensor/qwen-dumps/model_weights/model_"+std::to_string(idx)+"_input_4.bin").c_str(), "rb");
        fseek(fp, 0, SEEK_END);
        nSize = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        half_ptr k_bias_buffer = new uint16_t[512];
        fread(k_bias_buffer, 1, nSize/2, fp);
        std::cout << "k_bias: " << unsigned(k_bias_buffer[0]) << " " << unsigned(k_bias_buffer[1]) << std::endl;
        auto k_bias = Tensor(ov::element::Type_t::f16, ov::Shape({1, 512}), k_bias_buffer);
        weight_buffers.push_back(k_bias);
        fp = nullptr;
        r = fopen_s(&fp, ("C:/Users/SAS/kai/remote-tensor/qwen-dumps/model_weights/model_"+std::to_string(idx)+"_input_5.bin").c_str(), "rb");
        fseek(fp, 0, SEEK_END);
        nSize = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        half_ptr v_bias_buffer = new uint16_t[512];
        fread(v_bias_buffer, 1, nSize/2, fp);
        std::cout << "v_bias: " << unsigned(v_bias_buffer[0]) << " " << unsigned(v_bias_buffer[1]) << std::endl;
        auto v_bias = Tensor(ov::element::Type_t::f16, ov::Shape({1, 512}), v_bias_buffer);
        weight_buffers.push_back(v_bias);

        fp = nullptr;
        r = fopen_s(&fp, ("C:/Users/SAS/kai/remote-tensor/qwen-dumps/model_weights/model_"+std::to_string(idx)+"_input_8.bin").c_str(), "rb");
        fseek(fp, 0, SEEK_END);
        nSize = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        uint8_t* q_proj_weight_buffer = new uint8_t[3584 * 3584 / 2];
        fread(q_proj_weight_buffer, 1, nSize, fp);
        std::cout << "q_proj: "<< unsigned(q_proj_weight_buffer[0]) << " " << unsigned(q_proj_weight_buffer[1]) << std::endl;
        auto q_proj_weight = Tensor(ov::element::Type_t::i4, ov::Shape({3584, 3584}), q_proj_weight_buffer);
        weight_buffers.push_back(q_proj_weight);
        fp = nullptr;
        r = fopen_s(&fp, ("C:/Users/SAS/kai/remote-tensor/qwen-dumps/model_weights/model_"+std::to_string(idx)+"_input_9.bin").c_str(), "rb");
        fseek(fp, 0, SEEK_END);
        nSize = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        half_ptr q_proj_scale_buffer = new uint16_t[3584];
        fread(q_proj_scale_buffer, 1, nSize/2, fp);
        std::cout << "q_proj scale: " << unsigned(q_proj_scale_buffer[0]) << " " << unsigned(q_proj_scale_buffer[1]) << std::endl;
        auto q_proj_scale = Tensor(ov::element::Type_t::f16, ov::Shape({1, 3584}), q_proj_scale_buffer);
        weight_buffers.push_back(q_proj_scale);

        fp = nullptr;
        r = fopen_s(&fp, ("C:/Users/SAS/kai/remote-tensor/qwen-dumps/model_weights/model_"+std::to_string(idx)+"_input_10.bin").c_str(), "rb");
        fseek(fp, 0, SEEK_END);
        nSize = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        uint8_t* k_proj_weight_buffer = new uint8_t[512 * 3584 / 2];
        fread(k_proj_weight_buffer, 1, nSize, fp);
        auto k_proj_weight = Tensor(ov::element::Type_t::i4, ov::Shape({512, 3584}), k_proj_weight_buffer);
        weight_buffers.push_back(k_proj_weight);
        fp = nullptr;
        r = fopen_s(&fp, ("C:/Users/SAS/kai/remote-tensor/qwen-dumps/model_weights/model_"+std::to_string(idx)+"_input_11.bin").c_str(), "rb");
        fseek(fp, 0, SEEK_END);
        nSize = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        half_ptr k_proj_scale_buffer = new uint16_t[512];
        fread(k_proj_scale_buffer, 1, nSize/2, fp);
        auto k_proj_scale = Tensor(ov::element::Type_t::f16, ov::Shape({1, 512}), k_proj_scale_buffer);
        weight_buffers.push_back(k_proj_scale);

        fp = nullptr;
        r = fopen_s(&fp, ("C:/Users/SAS/kai/remote-tensor/qwen-dumps/model_weights/model_"+std::to_string(idx)+"_input_12.bin").c_str(), "rb");
        fseek(fp, 0, SEEK_END);
        nSize = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        uint8_t* v_proj_weight_buffer = new uint8_t[512 * 3584 / 2];
        fread(v_proj_weight_buffer, 1, nSize, fp);
        auto v_proj_weight = Tensor(ov::element::Type_t::i4, ov::Shape({512, 3584}), v_proj_weight_buffer);
        weight_buffers.push_back(v_proj_weight);
        fp = nullptr;
        r = fopen_s(&fp, ("C:/Users/SAS/kai/remote-tensor/qwen-dumps/model_weights/model_"+std::to_string(idx)+"_input_13.bin").c_str(), "rb");
        fseek(fp, 0, SEEK_END);
        nSize = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        half_ptr v_proj_scale_buffer = new uint16_t[512];
        fread(v_proj_scale_buffer, 1, nSize/2, fp);
        auto v_proj_scale = Tensor(ov::element::Type_t::f16, ov::Shape({1, 512}), v_proj_scale_buffer);
        weight_buffers.push_back(v_proj_scale);

        fp = nullptr;
        r = fopen_s(&fp, ("C:/Users/SAS/kai/remote-tensor/qwen-dumps/model_weights/model_"+std::to_string(idx)+"_input_14.bin").c_str(), "rb");
        fseek(fp, 0, SEEK_END);
        nSize = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        uint8_t* o_proj_weight_buffer = new uint8_t[3584 * 3584 / 2];
        fread(o_proj_weight_buffer, 1, nSize, fp);
        auto o_proj_weight = Tensor(ov::element::Type_t::i4, ov::Shape({3584, 3584}), o_proj_weight_buffer);
        weight_buffers.push_back(o_proj_weight);
        fp = nullptr;
        r = fopen_s(&fp, ("C:/Users/SAS/kai/remote-tensor/qwen-dumps/model_weights/model_"+std::to_string(idx)+"_input_15.bin").c_str(), "rb");
        fseek(fp, 0, SEEK_END);
        nSize = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        half_ptr o_proj_scale_buffer = new uint16_t[3584];
        fread(o_proj_scale_buffer, 1, nSize/2, fp);
        auto o_proj_scale = Tensor(ov::element::Type_t::f16, ov::Shape({1, 3584}), o_proj_scale_buffer);
        weight_buffers.push_back(o_proj_scale);

        fp = nullptr;
        r = fopen_s(&fp, ("C:/Users/SAS/kai/remote-tensor/qwen-dumps/model_weights/model_"+std::to_string(idx)+"_input_16.bin").c_str(), "rb");
        fseek(fp, 0, SEEK_END);
        nSize = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        uint8_t* gate_proj_weight_buffer = new uint8_t[18944 * 3584 / 2];
        fread(gate_proj_weight_buffer, 1, nSize, fp);
        auto gate_proj_weight = Tensor(ov::element::Type_t::i4, ov::Shape({18944, 3584}), gate_proj_weight_buffer);
        weight_buffers.push_back(gate_proj_weight);
        fp = nullptr;
        r = fopen_s(&fp, ("C:/Users/SAS/kai/remote-tensor/qwen-dumps/model_weights/model_"+std::to_string(idx)+"_input_17.bin").c_str(), "rb");
        fseek(fp, 0, SEEK_END);
        nSize = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        half_ptr gate_proj_scale_buffer = new uint16_t[18944];
        fread(gate_proj_scale_buffer, 1, nSize/2, fp);
        auto gate_proj_scale = Tensor(ov::element::Type_t::f16, ov::Shape({1, 18944}), gate_proj_scale_buffer);
        weight_buffers.push_back(gate_proj_scale);

        fp = nullptr;
        r = fopen_s(&fp, ("C:/Users/SAS/kai/remote-tensor/qwen-dumps/model_weights/model_"+std::to_string(idx)+"_input_18.bin").c_str(), "rb");
        fseek(fp, 0, SEEK_END);
        nSize = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        uint8_t* up_proj_weight_buffer = new uint8_t[18944 * 3584 / 2];
        fread(up_proj_weight_buffer, 1, nSize, fp);
        auto up_proj_weight = Tensor(ov::element::Type_t::i4, ov::Shape({18944, 3584}), up_proj_weight_buffer);
        weight_buffers.push_back(up_proj_weight);
        fp = nullptr;
        r = fopen_s(&fp, ("C:/Users/SAS/kai/remote-tensor/qwen-dumps/model_weights/model_"+std::to_string(idx)+"_input_19.bin").c_str(), "rb");
        fseek(fp, 0, SEEK_END);
        nSize = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        half_ptr up_proj_scale_buffer = new uint16_t[18944];
        fread(up_proj_scale_buffer, 1, nSize/2, fp);
        auto up_proj_scale = Tensor(ov::element::Type_t::f16, ov::Shape({1, 18944}), up_proj_scale_buffer);
        weight_buffers.push_back(up_proj_scale);

        fp = nullptr;
        r = fopen_s(&fp, ("C:/Users/SAS/kai/remote-tensor/qwen-dumps/model_weights/model_"+std::to_string(idx)+"_input_20.bin").c_str(), "rb");
        fseek(fp, 0, SEEK_END);
        nSize = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        uint8_t* down_proj_weight_buffer = new uint8_t[2 * 3584 * 9472 / 2];
        fread(down_proj_weight_buffer, 1, nSize, fp);
        auto down_proj_weight = Tensor(ov::element::Type_t::i4, ov::Shape({2, 3584, 9472}), down_proj_weight_buffer);
        weight_buffers.push_back(down_proj_weight);
        fp = nullptr;
        r = fopen_s(&fp, ("C:/Users/SAS/kai/remote-tensor/qwen-dumps/model_weights/model_"+std::to_string(idx)+"_input_21.bin").c_str(), "rb");
        fseek(fp, 0, SEEK_END);
        nSize = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        half_ptr down_proj_scale_buffer = new uint16_t[2 * 3584];
        fread(down_proj_scale_buffer, 1, nSize/2, fp);
        auto down_proj_scale = Tensor(ov::element::Type_t::f16, ov::Shape({2, 3584}), down_proj_scale_buffer);
        weight_buffers.push_back(down_proj_scale);

        layer_factory->setInputTensor(q_bias.get_tensor(), 3);
        layer_factory->setInputTensor(k_bias.get_tensor(), 4);
        layer_factory->setInputTensor(v_bias.get_tensor(), 5);
        layer_factory->setInputTensor(q_proj_weight.get_tensor(), 8);
        layer_factory->setInputTensor(q_proj_scale.get_tensor(), 9);
        layer_factory->setInputTensor(k_proj_weight.get_tensor(), 10);
        layer_factory->setInputTensor(k_proj_scale.get_tensor(), 11);
        layer_factory->setInputTensor(v_proj_weight.get_tensor(), 12);
        layer_factory->setInputTensor(v_proj_scale.get_tensor(), 13);
        layer_factory->setInputTensor(o_proj_weight.get_tensor(), 14);
        layer_factory->setInputTensor(o_proj_scale.get_tensor(), 15);
        layer_factory->setInputTensor(gate_proj_weight.get_tensor(), 16);
        layer_factory->setInputTensor(gate_proj_scale.get_tensor(), 17);
        layer_factory->setInputTensor(up_proj_weight.get_tensor(), 18);
        layer_factory->setInputTensor(up_proj_scale.get_tensor(), 19);
        layer_factory->setInputTensor(down_proj_weight.get_tensor(), 20);
        layer_factory->setInputTensor(down_proj_scale.get_tensor(), 21);
        layer_factory->setInputTensor(hidden_buffer, 0);
        layer_factory->setInputTensor(attention_mask, 1);
        layer_factory->setInputTensor(position_id, 2);
        half_ptr k_cache = new uint16_t[1 * 4 * 1023 * 128];
        half_ptr v_cache = new uint16_t[1 * 4 * 1023 * 128];
        memset(k_cache, 0, 4 * 1023 * 128 * sizeof(uint16_t));
        memset(v_cache, 0, 4 * 1023 * 128 * sizeof(uint16_t));
        layer_factory->setInputTensor(k_cache, 6);
        layer_factory->setInputTensor(v_cache, 7);
        k_caches.push_back(k_cache);
        v_caches.push_back(v_cache);

        // half_ptr hidden_buffer = new uint16_t[3584];
        // auto hidden_buffer = layer_factory->createRemoteOutputTensor(0);
        // hidden_buffers.push_back(hidden_buffer);
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

    // LM-Head
    auto lm_head_factory = std::make_shared<ModelFactory>("NPU");
    lm_head_factory->create_ov_model("C:/Users/SAS/kai/remote-tensor/qwen-dumps/lm_head_new.xml");
    // Read int8 weight
    fp = nullptr;
    r = fopen_s(&fp, "C:/Users/SAS/kai/remote-tensor/qwen-dumps/model_weights/model_lm_head_input_1.bin", "rb");
    fseek(fp, 0, SEEK_END);
    nSize = ftell(fp);
    std::cout << nSize << std::endl;
    fseek(fp, 0, SEEK_SET);
    uint8_t* lm_head_weight_buffer = new uint8_t[152064 * 3584];
    fread(lm_head_weight_buffer, 1, nSize, fp);
    auto lm_head_weight = Tensor(ov::element::Type_t::i8, ov::Shape({152064, 3584}), lm_head_weight_buffer);
    // std::vector<uint8_t> lm_weight(lm_head_weight_buffer, lm_head_weight_buffer + 152064 * 3584);
    // std::cout << unsigned(lm_weight[0]) << " " << unsigned(lm_weight[2]) << std::endl;

    // Read fp16 bias
    fp = nullptr;
    r = fopen_s(&fp, "C:/Users/SAS/kai/remote-tensor/qwen-dumps/model_weights/model_lm_head_input_2.bin", "rb");
    fseek(fp, 0, SEEK_END);
    nSize = ftell(fp);
    std::cout << nSize << std::endl;
    fseek(fp, 0, SEEK_SET);
    half_ptr lm_head_scale_buffer = new uint16_t[152064];
    fread(lm_head_scale_buffer, 1, nSize/2, fp);
    auto lm_head_scale = Tensor(ov::element::Type_t::f16, ov::Shape({1, 152064}), lm_head_scale_buffer);
    // std::vector<uint16_t> lm_bias(lm_head_bias_buffer, lm_head_bias_buffer + 152064);
    // std::cout << (unsigned int)lm_bias[0] << " " << (unsigned int)lm_bias[2] << std::endl;

    float* logits_buffer = new float[152064];

    lm_head_factory->setInputTensor(hidden_buffer, 0);
    lm_head_factory->setInputTensor(lm_head_weight.get_tensor(), 1);
    lm_head_factory->setInputTensor(lm_head_scale.get_tensor(), 2);
    lm_head_factory->setOutputTensor(logits_buffer, 0);

    const size_t N = 20;
    std::cout << "Run inference on " << N << " workloads" << std::endl;
    auto start = high_resolution_clock::now();
    for (auto idx = 0; idx < N; idx++) {
        embedding_factory->run();
        for (int idx = 0; idx < num_layers; idx++) {
            decoder_layers[idx]->run();
        }
        lm_head_factory->run();
        // std::cout << embed_buffer[0] << " " << embed_buffer[1] << std::endl;
        // std::cout << hidden_buffers[0][0] << " " << hidden_buffers[0][1] << std::endl;
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    std::cout << "Average time used " << (double)duration.count()/N << std::endl;
    // std::vector<float> logits(logits_buffer, logits_buffer + 152064);
    std::cout << logits_buffer[0] << " " << logits_buffer[1] << std::endl;

    std::cout << "Inference done" << std::endl;
    system("pause");
    return 0;
}