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
    std::vector<half_ptr> hidden_buffers;

    uint32_t* input_id = new uint32_t[1];
    input_id[0] = 100;
    std::cout << input_id[0] << std::endl;
    uint64_t* attention_mask = new uint64_t[1024];
    int64_t minVal = std::numeric_limits<int64_t>::min();
    for(int i = 0; i < 1024; i++){
        attention_mask[i] = minVal;
    }
    uint64_t* position_id = new uint64_t[1];
    position_id[0] = 0;

    // Embedding
    auto embedding_factory = std::make_shared<ModelFactory>("NPU");
    embedding_factory->create_ov_model("C:/Users/Administrator/kai/remote-tensor/qwen-dumps/embedding_new.xml");

    half_ptr embed_buffer = new uint16_t[3584];
    hidden_buffers.push_back(embed_buffer);
    // auto embed_buffer = embedding_factory->createRemoteOutputTensor(0);
    embedding_factory->setInputTensor(input_id, 0);
    embedding_factory->setOutputTensor(embed_buffer, 0);

    // Decoder
    std::vector<std::shared_ptr<ModelFactory>> decoder_layers;
    std::vector<half_ptr> k_caches;
    std::vector<half_ptr> v_caches;
    std::vector<half_ptr> k_results;
    std::vector<half_ptr> v_results;
    std::vector<half_ptr> half_buffers;
    std::vector<uint8_t*> uint8_buffers;
    for (int idx = 0; idx < num_layers; idx++) {
        std::cout << "Loading layer " << idx << std::endl;
        auto layer_factory = std::make_shared<ModelFactory>("NPU");
        layer_factory->create_ov_model("C:/Users/Administrator/kai/remote-tensor/qwen-dumps/decoder_layer_"+std::to_string(idx)+"_new.xml");
        r = fopen_s(&fp, ("C:/Users/Administrator/kai/remote-tensor/qwen-dumps/model_weights/model_"+std::to_string(idx)+"_input_3.bin").c_str(), "rb");
        fseek(fp, 0, SEEK_END);
        nSize = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        half_ptr q_bias_buffer = new uint16_t[3584];
        fread(q_bias_buffer, 1, nSize/2, fp);
        half_buffers.push_back(q_bias_buffer);
        std::cout << "q_bias: " << unsigned(q_bias_buffer[0]) << " " << unsigned(q_bias_buffer[1]) << std::endl;
        r = fopen_s(&fp, ("C:/Users/Administrator/kai/remote-tensor/qwen-dumps/model_weights/model_"+std::to_string(idx)+"_input_4.bin").c_str(), "rb");
        fseek(fp, 0, SEEK_END);
        nSize = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        half_ptr k_bias_buffer = new uint16_t[512];
        fread(k_bias_buffer, 1, nSize/2, fp);
        half_buffers.push_back(k_bias_buffer);
        std::cout << "k_bias: " << unsigned(k_bias_buffer[0]) << " " << unsigned(k_bias_buffer[1]) << std::endl;
        r = fopen_s(&fp, ("C:/Users/Administrator/kai/remote-tensor/qwen-dumps/model_weights/model_"+std::to_string(idx)+"_input_5.bin").c_str(), "rb");
        fseek(fp, 0, SEEK_END);
        nSize = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        half_ptr v_bias_buffer = new uint16_t[512];
        fread(v_bias_buffer, 1, nSize/2, fp);
        half_buffers.push_back(v_bias_buffer);
        std::cout << "v_bias: " << unsigned(v_bias_buffer[0]) << " " << unsigned(v_bias_buffer[1]) << std::endl;

        r = fopen_s(&fp, ("C:/Users/Administrator/kai/remote-tensor/qwen-dumps/model_weights/model_"+std::to_string(idx)+"_input_8.bin").c_str(), "rb");
        fseek(fp, 0, SEEK_END);
        nSize = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        uint8_t* q_proj_weight_buffer = new uint8_t[3584 * 3584 / 2];
        fread(q_proj_weight_buffer, 1, nSize, fp);
        uint8_buffers.push_back(q_proj_weight_buffer);
        std::cout << "q_proj: "<< unsigned(q_proj_weight_buffer[0]) << " " << unsigned(q_proj_weight_buffer[1]) << std::endl;
        r = fopen_s(&fp, ("C:/Users/Administrator/kai/remote-tensor/qwen-dumps/model_weights/model_"+std::to_string(idx)+"_input_9.bin").c_str(), "rb");
        fseek(fp, 0, SEEK_END);
        nSize = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        half_ptr q_proj_scale_buffer = new uint16_t[3584];
        fread(q_proj_scale_buffer, 1, nSize/2, fp);
        half_buffers.push_back(q_proj_scale_buffer);
        std::cout << "q_proj scale: " << unsigned(q_proj_scale_buffer[0]) << " " << unsigned(q_proj_scale_buffer[1]) << std::endl;

        r = fopen_s(&fp, ("C:/Users/Administrator/kai/remote-tensor/qwen-dumps/model_weights/model_"+std::to_string(idx)+"_input_10.bin").c_str(), "rb");
        fseek(fp, 0, SEEK_END);
        nSize = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        uint8_t* k_proj_weight_buffer = new uint8_t[512 * 3584 / 2];
        fread(k_proj_weight_buffer, 1, nSize, fp);
        uint8_buffers.push_back(k_proj_weight_buffer);
        r = fopen_s(&fp, ("C:/Users/Administrator/kai/remote-tensor/qwen-dumps/model_weights/model_"+std::to_string(idx)+"_input_11.bin").c_str(), "rb");
        fseek(fp, 0, SEEK_END);
        nSize = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        half_ptr k_proj_scale_buffer = new uint16_t[512];
        fread(k_proj_scale_buffer, 1, nSize/2, fp);
        half_buffers.push_back(k_proj_scale_buffer);

        r = fopen_s(&fp, ("C:/Users/Administrator/kai/remote-tensor/qwen-dumps/model_weights/model_"+std::to_string(idx)+"_input_12.bin").c_str(), "rb");
        fseek(fp, 0, SEEK_END);
        nSize = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        uint8_t* v_proj_weight_buffer = new uint8_t[512 * 3584 / 2];
        uint8_buffers.push_back(v_proj_weight_buffer);
        fread(v_proj_weight_buffer, 1, nSize, fp);
        r = fopen_s(&fp, ("C:/Users/Administrator/kai/remote-tensor/qwen-dumps/model_weights/model_"+std::to_string(idx)+"_input_13.bin").c_str(), "rb");
        fseek(fp, 0, SEEK_END);
        nSize = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        half_ptr v_proj_scale_buffer = new uint16_t[512];
        fread(v_proj_scale_buffer, 1, nSize/2, fp);
        half_buffers.push_back(v_proj_scale_buffer);

        r = fopen_s(&fp, ("C:/Users/Administrator/kai/remote-tensor/qwen-dumps/model_weights/model_"+std::to_string(idx)+"_input_14.bin").c_str(), "rb");
        fseek(fp, 0, SEEK_END);
        nSize = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        uint8_t* o_proj_weight_buffer = new uint8_t[3584 * 3584 / 2];
        fread(o_proj_weight_buffer, 1, nSize, fp);
        uint8_buffers.push_back(o_proj_weight_buffer);
        r = fopen_s(&fp, ("C:/Users/Administrator/kai/remote-tensor/qwen-dumps/model_weights/model_"+std::to_string(idx)+"_input_15.bin").c_str(), "rb");
        fseek(fp, 0, SEEK_END);
        nSize = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        half_ptr o_proj_scale_buffer = new uint16_t[3584];
        fread(o_proj_scale_buffer, 1, nSize/2, fp);
        half_buffers.push_back(o_proj_scale_buffer);

        r = fopen_s(&fp, ("C:/Users/Administrator/kai/remote-tensor/qwen-dumps/model_weights/model_"+std::to_string(idx)+"_input_16.bin").c_str(), "rb");
        fseek(fp, 0, SEEK_END);
        nSize = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        uint8_t* gate_proj_weight_buffer = new uint8_t[18944 * 3584 / 2];
        fread(gate_proj_weight_buffer, 1, nSize, fp);
        uint8_buffers.push_back(gate_proj_weight_buffer);
        r = fopen_s(&fp, ("C:/Users/Administrator/kai/remote-tensor/qwen-dumps/model_weights/model_"+std::to_string(idx)+"_input_17.bin").c_str(), "rb");
        fseek(fp, 0, SEEK_END);
        nSize = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        half_ptr gate_proj_scale_buffer = new uint16_t[18944];
        fread(gate_proj_scale_buffer, 1, nSize/2, fp);
        half_buffers.push_back(gate_proj_scale_buffer);

        r = fopen_s(&fp, ("C:/Users/Administrator/kai/remote-tensor/qwen-dumps/model_weights/model_"+std::to_string(idx)+"_input_18.bin").c_str(), "rb");
        fseek(fp, 0, SEEK_END);
        nSize = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        uint8_t* up_proj_weight_buffer = new uint8_t[18944 * 3584 / 2];
        fread(up_proj_weight_buffer, 1, nSize, fp);
        uint8_buffers.push_back(up_proj_weight_buffer);
        r = fopen_s(&fp, ("C:/Users/Administrator/kai/remote-tensor/qwen-dumps/model_weights/model_"+std::to_string(idx)+"_input_19.bin").c_str(), "rb");
        fseek(fp, 0, SEEK_END);
        nSize = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        half_ptr up_proj_scale_buffer = new uint16_t[18944];
        fread(up_proj_scale_buffer, 1, nSize/2, fp);
        half_buffers.push_back(up_proj_scale_buffer);

        r = fopen_s(&fp, ("C:/Users/Administrator/kai/remote-tensor/qwen-dumps/model_weights/model_"+std::to_string(idx)+"_input_20.bin").c_str(), "rb");
        fseek(fp, 0, SEEK_END);
        nSize = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        uint8_t* down_proj_weight_buffer = new uint8_t[2 * 3584 * 9472 / 2];
        fread(down_proj_weight_buffer, 1, nSize, fp);
        uint8_buffers.push_back(down_proj_weight_buffer);
        r = fopen_s(&fp, ("C:/Users/Administrator/kai/remote-tensor/qwen-dumps/model_weights/model_"+std::to_string(idx)+"_input_21.bin").c_str(), "rb");
        fseek(fp, 0, SEEK_END);
        nSize = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        half_ptr down_proj_scale_buffer = new uint16_t[2 * 3584];
        fread(down_proj_scale_buffer, 1, nSize/2, fp);
        half_buffers.push_back(down_proj_scale_buffer);

        layer_factory->setInputTensor(q_bias_buffer, 3);
        layer_factory->setInputTensor(k_bias_buffer, 4);
        layer_factory->setInputTensor(v_bias_buffer, 5);
        layer_factory->setInputTensor(q_proj_weight_buffer, 8);
        layer_factory->setInputTensor(q_proj_scale_buffer, 9);
        layer_factory->setInputTensor(k_proj_weight_buffer, 10);
        layer_factory->setInputTensor(k_proj_scale_buffer, 11);
        layer_factory->setInputTensor(v_proj_weight_buffer, 12);
        layer_factory->setInputTensor(v_proj_scale_buffer, 13);
        layer_factory->setInputTensor(o_proj_weight_buffer, 14);
        layer_factory->setInputTensor(o_proj_scale_buffer, 15);
        layer_factory->setInputTensor(gate_proj_weight_buffer, 16);
        layer_factory->setInputTensor(gate_proj_scale_buffer, 17);
        layer_factory->setInputTensor(up_proj_weight_buffer, 18);
        layer_factory->setInputTensor(up_proj_scale_buffer, 19);
        layer_factory->setInputTensor(down_proj_weight_buffer, 20);
        layer_factory->setInputTensor(down_proj_scale_buffer, 21);
        layer_factory->setInputTensor(hidden_buffers[idx], 0);
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

        half_ptr hidden_buffer = new uint16_t[3584];
        hidden_buffers.push_back(hidden_buffer);
        // auto hidden_buffer = layer_factory->createRemoteOutputTensor(0);
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
    lm_head_factory->create_ov_model("C:/Users/Administrator/kai/remote-tensor/qwen-dumps/lm_head_new.xml");
    // Read int8 weight
    r = fopen_s(&fp, "C:/Users/Administrator/kai/remote-tensor/qwen-dumps/model_weights/model_lm_head_input_1.bin", "rb");
    fseek(fp, 0, SEEK_END);
    nSize = ftell(fp);
    std::cout << nSize << std::endl;
    fseek(fp, 0, SEEK_SET);
    uint8_t* lm_head_weight_buffer = new uint8_t[152064 * 3584];
    fread(lm_head_weight_buffer, 1, nSize, fp);
    // std::vector<uint8_t> lm_weight(lm_head_weight_buffer, lm_head_weight_buffer + 152064 * 3584);
    // std::cout << unsigned(lm_weight[0]) << " " << unsigned(lm_weight[2]) << std::endl;

    // Read fp16 bias
    fp = nullptr;
    r = fopen_s(&fp, "C:/Users/Administrator/kai/remote-tensor/qwen-dumps/model_weights/model_lm_head_input_2.bin", "rb");
    fseek(fp, 0, SEEK_END);
    nSize = ftell(fp);
    std::cout << nSize << std::endl;
    fseek(fp, 0, SEEK_SET);
    half_ptr lm_head_scale_buffer = new uint16_t[152064];
    fread(lm_head_scale_buffer, 1, nSize/2, fp);
    // std::vector<uint16_t> lm_bias(lm_head_bias_buffer, lm_head_bias_buffer + 152064);
    // std::cout << (unsigned int)lm_bias[0] << " " << (unsigned int)lm_bias[2] << std::endl;

    // half_ptr hidden_buffer = new uint16_t[3584];
    // memset(hidden_buffer, 0, 3584 * sizeof(uint16_t));
    float* logits_buffer = new float[152064];

    lm_head_factory->setInputTensor(hidden_buffers.back(), 0);
    lm_head_factory->setInputTensor(lm_head_weight_buffer, 1);
    lm_head_factory->setInputTensor(lm_head_scale_buffer, 2);
    lm_head_factory->setOutputTensor(logits_buffer, 0);

    const size_t N = 10;
    std::cout << "Run inference on " << N << " workloads" << std::endl;
    auto start = high_resolution_clock::now();
    for (auto idx = 0; idx < N; idx++) {
        embedding_factory->run();
        for (int idx = 0; idx < num_layers; idx++) {
            decoder_layers[idx]->run();
        }
        lm_head_factory->run();
        std::cout << embed_buffer[0] << " " << embed_buffer[1] << std::endl;
        std::cout << hidden_buffers[0][0] << " " << hidden_buffers[0][1] << std::endl;
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    std::cout << "Average time used " << (double)duration.count()/N << std::endl;
    std::vector<float> logits(logits_buffer, logits_buffer + 152064);
    std::cout << logits[0] << " " << logits[2] << std::endl;


    // const size_t batch = 128, inC = 256, outC = 512, N = 10000;

    // std::cout << "Create a ModelFactory" << std::endl;
    // auto factory = std::make_shared<ModelFactory>("NPU");

    // // create parameter
    // auto input = factory->parameter({batch, inC}, ov::element::f16);
    // auto weights = factory->parameter({outC, inC}, ov::element::f16);
    // auto bias = factory->parameter({1, outC}, ov::element::f16);

    // // create matmul
    // auto matmul = factory->matmul(input, weights);
    // auto matmul_bias = factory->eltwise_add(matmul, bias);
    // factory->result(matmul_bias);

    // // Compile the model
    // factory->compile();

    // // Save OV model
    // // std::cout << "Saving model to matmul.xml" << std::endl;
    // // factory->saveModel("matmul.xml");

    // half_ptr input_buffer = new uint16_t[batch * inC];
    // half_ptr weights_buffer = new uint16_t[outC * inC];
    // half_ptr bias_buffer = new uint16_t[outC];
    // memset(input_buffer, 0, batch * inC * sizeof(uint16_t));
    // memset(weights_buffer, 0, outC * inC * sizeof(uint16_t));
    // memset(bias_buffer, 0, outC * sizeof(uint16_t));
    // // half_ptr output_buffer = new uint16_t[batch * outC];
    // // memset(output_buffer, 0, batch * outC * sizeof(uint16_t));

    // std::cout << "Creating a remote tensor" << std::endl;
    // // auto input_buffer = factory->createRemoteInputTensor(0);
    // // auto weights_buffer = factory->createRemoteInputTensor(1);
    // // auto bias_buffer = factory->createRemoteInputTensor(2);
    // // std::memset(input_buffer.get(), 0, input_buffer.get_byte_size());
    // // std::memset(weights_buffer.get(), 0, weights_buffer.get_byte_size());
    // // std::memset(bias_buffer.get(), 0, bias_buffer.get_byte_size());
    // auto output_buffer = factory->createRemoteOutputTensor(0);
    // std::memset(output_buffer.get(), 0, output_buffer.get_byte_size());

    // factory->setInputTensor(input_buffer, 0);
    // factory->setInputTensor(weights_buffer, 1);
    // factory->setInputTensor(bias_buffer, 2);
    // factory->setOutputTensor(output_buffer, 0);

    // const size_t inC2 = 512, outC2 = 1024;
    // auto factory2 = std::make_shared<ModelFactory>("NPU");
    // auto input2 = factory2->parameter({batch, inC2}, ov::element::f16);
    // auto weights2 = factory2->parameter({outC2, inC2}, ov::element::f16);
    // auto bias2 = factory2->parameter({1, outC2}, ov::element::f16);
    // auto matmul2 = factory2->matmul(input2, weights2);
    // auto matmul_bias2 = factory2->eltwise_add(matmul2, bias2);
    // factory2->result(matmul_bias2);
    // factory2->compile();

    // half_ptr weights_buffer2 = new uint16_t[outC2 * inC2];
    // half_ptr bias_buffer2 = new uint16_t[outC2];
    // half_ptr output_buffer2 = new uint16_t[batch * outC2];
    // memset(weights_buffer2, 0, outC2 * inC2 * sizeof(uint16_t));
    // memset(output_buffer2, 0, batch * outC2 * sizeof(uint16_t));
    // memset(bias_buffer2, 0, outC2 * sizeof(uint16_t));

    // // auto weights_buffer2 = factory2->createRemoteInputTensor(1);
    // // auto bias_buffer2 = factory2->createRemoteInputTensor(2);
    // // auto output_buffer2 = factory2->createRemoteOutputTensor(0);
    // // std::memset(weights_buffer2.get(), 0, weights_buffer2.get_byte_size());
    // // std::memset(bias_buffer2.get(), 0, bias_buffer2.get_byte_size());
    // // std::memset(output_buffer2.get(), 0, output_buffer2.get_byte_size());
    // factory2->setInputTensor(output_buffer, 0);
    // factory2->setInputTensor(weights_buffer2, 1);
    // factory2->setInputTensor(bias_buffer2, 2);
    // factory2->setOutputTensor(output_buffer2, 0);

    // // Run inference
    // std::cout << "Run inference on " << N << " workloads" << std::endl;
    // auto start = high_resolution_clock::now();
    // for (auto idx = 0; idx < N; idx++) {
    //     factory->run();
    //     factory2->run();
    // }
    // auto stop = high_resolution_clock::now();
    // auto duration = duration_cast<milliseconds>(stop - start);
    // std::cout << "Average time used " << (double)duration.count()/N << std::endl;

    std::cout << "Inference done" << std::endl;
    system("pause");
    return 0;
}