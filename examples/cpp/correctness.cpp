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

    uint32_t* input_id = new uint32_t[1];
    input_id[0] = 100;
    uint64_t* attention_mask = new uint64_t[1024];
    int64_t minVal = std::numeric_limits<int64_t>::min();
    for(int i = 0; i < 1024; i++){
        attention_mask[i] = minVal;
    }
    uint64_t* position_id = new uint64_t[1];
    position_id[0] = 0;
    attention_mask[0] = 0;
    attention_mask[1023] = 0;

    auto embedding_factory = std::make_shared<ModelFactory>("NPU");
    embedding_factory->create_ov_model("D:/kai/remote-tensor/qwen-dumps/embedding_new.xml");
    auto hidden_buffer = embedding_factory->createRemoteOutputTensor(0);
    embedding_factory->setInputTensor(input_id, 0);
    embedding_factory->setOutputTensor(hidden_buffer, 0);

    int idx = 0;
    std::cout << "Loading layer " << idx << std::endl;
    auto layer_factory = std::make_shared<ModelFactory>("NPU");
    layer_factory->create_ov_model("D:/kai/remote-tensor/qwen-dumps/decoder_layer_"+std::to_string(idx)+"_new.xml");
    layer_factory->setInputTensor(hidden_buffer, 0);
    layer_factory->setInputTensor(attention_mask, 1);
    layer_factory->setInputTensor(position_id, 2);

    r = fopen_s(&fp, ("D:/kai/remote-tensor/qwen-dumps/model_weights/model_"+std::to_string(idx)+"_input_3.bin").c_str(), "rb");
    fseek(fp, 0, SEEK_END);
    nSize = ftell(fp);
    // std::cout << nSize << std::endl;
    fseek(fp, 0, SEEK_SET);
    half_ptr input_layernorm_buffer = new uint16_t[3584];
    fread(input_layernorm_buffer, 2, nSize/2, fp);
    // std::cout << unsigned(input_layernorm_buffer[0]) << " " << unsigned(input_layernorm_buffer[1]) << std::endl;
    // std::cout << unsigned(input_layernorm_buffer[3581]) << " " << unsigned(input_layernorm_buffer[3583]) << std::endl;
    auto input_layer_norm = Tensor(ov::element::Type_t::f16, ov::Shape({1, 3584}), input_layernorm_buffer);
    layer_factory->setInputTensor(input_layer_norm.get_tensor(), 3);

    fp = nullptr;
    r = fopen_s(&fp, ("D:/kai/remote-tensor/qwen-dumps/model_weights/model_"+std::to_string(idx)+"_input_4.bin").c_str(), "rb");
    fseek(fp, 0, SEEK_END);
    nSize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    half_ptr output_layernorm_buffer = new uint16_t[3584];
    fread(output_layernorm_buffer, 2, nSize/2, fp);
    auto output_layer_norm = Tensor(ov::element::Type_t::f16, ov::Shape({1, 3584}), output_layernorm_buffer);
    layer_factory->setInputTensor(output_layer_norm.get_tensor(), 4);

    fp = nullptr;
    r = fopen_s(&fp, ("D:/kai/remote-tensor/qwen-dumps/model_weights/model_"+std::to_string(idx)+"_input_5.bin").c_str(), "rb");
    fseek(fp, 0, SEEK_END);
    nSize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    half_ptr q_bias_buffer = new uint16_t[3584];
    fread(q_bias_buffer, 2, nSize/2, fp);
    auto q_bias = Tensor(ov::element::Type_t::f16, ov::Shape({3584}), q_bias_buffer);
    layer_factory->setInputTensor(q_bias.get_tensor(), 5);

    fp = nullptr;
    r = fopen_s(&fp, ("D:/kai/remote-tensor/qwen-dumps/model_weights/model_"+std::to_string(idx)+"_input_6.bin").c_str(), "rb");
    fseek(fp, 0, SEEK_END);
    nSize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    half_ptr k_bias_buffer = new uint16_t[512];
    fread(k_bias_buffer, 2, nSize/2, fp);
    auto k_bias = Tensor(ov::element::Type_t::f16, ov::Shape({512}), k_bias_buffer);
    layer_factory->setInputTensor(k_bias.get_tensor(), 6);

    fp = nullptr;
    r = fopen_s(&fp, ("D:/kai/remote-tensor/qwen-dumps/model_weights/model_"+std::to_string(idx)+"_input_7.bin").c_str(), "rb");
    fseek(fp, 0, SEEK_END);
    nSize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    half_ptr v_bias_buffer = new uint16_t[512];
    fread(v_bias_buffer, 2, nSize/2, fp);
    auto v_bias = Tensor(ov::element::Type_t::f16, ov::Shape({512}), v_bias_buffer);
    layer_factory->setInputTensor(v_bias.get_tensor(), 7);

    half_ptr k_cache = new uint16_t[1 * 4 * 1023 * 128];
    half_ptr v_cache = new uint16_t[1 * 4 * 128 * 1023];
    memset(k_cache, 0, 4 * 1023 * 128 * sizeof(uint16_t));
    memset(v_cache, 0, 4 * 1023 * 128 * sizeof(uint16_t));
    layer_factory->setInputTensor(k_cache, 8);
    layer_factory->setInputTensor(v_cache, 9);

    fp = nullptr;
    r = fopen_s(&fp, ("D:/kai/remote-tensor/qwen-dumps/model_weights/model_"+std::to_string(idx)+"_input_10.bin").c_str(), "rb");
    fseek(fp, 0, SEEK_END);
    nSize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    uint8_t* q_proj_weight_buffer = new uint8_t[28 * 3584 * 128 / 2];
    fread(q_proj_weight_buffer, 1, nSize, fp);
    auto q_proj_weight = Tensor(ov::element::Type_t::i4, ov::Shape({28, 3584, 128}), q_proj_weight_buffer);
    layer_factory->setInputTensor(q_proj_weight.get_tensor(), 10);

    fp = nullptr;
    r = fopen_s(&fp, ("D:/kai/remote-tensor/qwen-dumps/model_weights/model_"+std::to_string(idx)+"_input_11.bin").c_str(), "rb");
    fseek(fp, 0, SEEK_END);
    nSize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    half_ptr q_proj_scale_buffer = new uint16_t[28 * 3584];
    fread(q_proj_scale_buffer, 2, nSize/2, fp);
    auto q_proj_scale = Tensor(ov::element::Type_t::f16, ov::Shape({28, 1, 3584}), q_proj_scale_buffer);
    layer_factory->setInputTensor(q_proj_scale.get_tensor(), 11);

    fp = nullptr;
    r = fopen_s(&fp, ("D:/kai/remote-tensor/qwen-dumps/model_weights/model_"+std::to_string(idx)+"_input_12.bin").c_str(), "rb");
    fseek(fp, 0, SEEK_END);
    nSize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    uint8_t* k_proj_weight_buffer = new uint8_t[28 * 512 * 128 / 2];
    fread(k_proj_weight_buffer, 1, nSize, fp);
    auto k_proj_weight = Tensor(ov::element::Type_t::i4, ov::Shape({28, 512, 128}), k_proj_weight_buffer);
    layer_factory->setInputTensor(k_proj_weight.get_tensor(), 12);

    fp = nullptr;
    r = fopen_s(&fp, ("D:/kai/remote-tensor/qwen-dumps/model_weights/model_"+std::to_string(idx)+"_input_13.bin").c_str(), "rb");
    fseek(fp, 0, SEEK_END);
    nSize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    half_ptr k_proj_scale_buffer = new uint16_t[28 * 512];
    fread(k_proj_scale_buffer, 2, nSize/2, fp);
    auto k_proj_scale = Tensor(ov::element::Type_t::f16, ov::Shape({28, 1, 512}), k_proj_scale_buffer);
    layer_factory->setInputTensor(k_proj_scale.get_tensor(), 13);

    fp = nullptr;
    r = fopen_s(&fp, ("D:/kai/remote-tensor/qwen-dumps/model_weights/model_"+std::to_string(idx)+"_input_14.bin").c_str(), "rb");
    fseek(fp, 0, SEEK_END);
    nSize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    uint8_t* v_proj_weight_buffer = new uint8_t[28 * 512 * 128 / 2];
    fread(v_proj_weight_buffer, 1, nSize, fp);
    auto v_proj_weight = Tensor(ov::element::Type_t::i4, ov::Shape({28, 512, 128}), v_proj_weight_buffer);
    layer_factory->setInputTensor(v_proj_weight.get_tensor(), 14);

    fp = nullptr;
    r = fopen_s(&fp, ("D:/kai/remote-tensor/qwen-dumps/model_weights/model_"+std::to_string(idx)+"_input_15.bin").c_str(), "rb");
    fseek(fp, 0, SEEK_END);
    nSize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    half_ptr v_proj_scale_buffer = new uint16_t[28 * 512];
    fread(v_proj_scale_buffer, 2, nSize/2, fp);
    auto v_proj_scale = Tensor(ov::element::Type_t::f16, ov::Shape({28, 1, 512}), v_proj_scale_buffer);
    layer_factory->setInputTensor(v_proj_scale.get_tensor(), 15);

    fp = nullptr;
    r = fopen_s(&fp, ("D:/kai/remote-tensor/qwen-dumps/model_weights/model_"+std::to_string(idx)+"_input_16.bin").c_str(), "rb");
    fseek(fp, 0, SEEK_END);
    nSize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    uint8_t* o_proj_weight_buffer = new uint8_t[28 * 3584 * 128 / 2];
    fread(o_proj_weight_buffer, 1, nSize, fp);
    auto o_proj_weight = Tensor(ov::element::Type_t::i4, ov::Shape({28, 3584, 128}), o_proj_weight_buffer);
    layer_factory->setInputTensor(o_proj_weight.get_tensor(), 16);

    fp = nullptr;
    r = fopen_s(&fp, ("D:/kai/remote-tensor/qwen-dumps/model_weights/model_"+std::to_string(idx)+"_input_17.bin").c_str(), "rb");
    fseek(fp, 0, SEEK_END);
    nSize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    half_ptr o_proj_scale_buffer = new uint16_t[28 * 3584];
    fread(o_proj_scale_buffer, 2, nSize/2, fp);
    auto o_proj_scale = Tensor(ov::element::Type_t::f16, ov::Shape({28, 1, 3584}), o_proj_scale_buffer);
    layer_factory->setInputTensor(o_proj_scale.get_tensor(), 17);

    fp = nullptr;
    r = fopen_s(&fp, ("D:/kai/remote-tensor/qwen-dumps/model_weights/model_"+std::to_string(idx)+"_input_18.bin").c_str(), "rb");
    fseek(fp, 0, SEEK_END);
    nSize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    uint8_t* gate_proj_weight_buffer = new uint8_t[28 * 18944 * 128 / 2];
    fread(gate_proj_weight_buffer, 1, nSize, fp);
    auto gate_proj_weight = Tensor(ov::element::Type_t::i4, ov::Shape({28, 18944, 128}), gate_proj_weight_buffer);
    layer_factory->setInputTensor(gate_proj_weight.get_tensor(), 18);

    fp = nullptr;
    r = fopen_s(&fp, ("D:/kai/remote-tensor/qwen-dumps/model_weights/model_"+std::to_string(idx)+"_input_19.bin").c_str(), "rb");
    fseek(fp, 0, SEEK_END);
    nSize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    half_ptr gate_proj_scale_buffer = new uint16_t[28 * 18944];
    fread(gate_proj_scale_buffer, 2, nSize/2, fp);
    auto gate_proj_scale = Tensor(ov::element::Type_t::f16, ov::Shape({28, 1, 18944}), gate_proj_scale_buffer);
    layer_factory->setInputTensor(gate_proj_scale.get_tensor(), 19);

    fp = nullptr;
    r = fopen_s(&fp, ("D:/kai/remote-tensor/qwen-dumps/model_weights/model_"+std::to_string(idx)+"_input_20.bin").c_str(), "rb");
    fseek(fp, 0, SEEK_END);
    nSize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    uint8_t* up_proj_weight_buffer = new uint8_t[28 * 18944 * 128 / 2];
    fread(up_proj_weight_buffer, 1, nSize, fp);
    auto up_proj_weight = Tensor(ov::element::Type_t::i4, ov::Shape({28, 18944, 128}), up_proj_weight_buffer);
    layer_factory->setInputTensor(up_proj_weight.get_tensor(), 20);

    fp = nullptr;
    r = fopen_s(&fp, ("D:/kai/remote-tensor/qwen-dumps/model_weights/model_"+std::to_string(idx)+"_input_21.bin").c_str(), "rb");
    fseek(fp, 0, SEEK_END);
    nSize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    half_ptr up_proj_scale_buffer = new uint16_t[28 * 18944];
    fread(up_proj_scale_buffer, 2, nSize/2, fp);
    auto up_proj_scale = Tensor(ov::element::Type_t::f16, ov::Shape({28, 1, 18944}), up_proj_scale_buffer);
    layer_factory->setInputTensor(up_proj_scale.get_tensor(), 21);

    fp = nullptr;
    r = fopen_s(&fp, ("D:/kai/remote-tensor/qwen-dumps/model_weights/model_"+std::to_string(idx)+"_input_22.bin").c_str(), "rb");
    fseek(fp, 0, SEEK_END);
    nSize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    uint8_t* down_proj_weight_buffer = new uint8_t[148 * 3584 * 128 / 2];
    fread(down_proj_weight_buffer, 1, nSize, fp);
    auto down_proj_weight = Tensor(ov::element::Type_t::i4, ov::Shape({148, 3584, 128}), down_proj_weight_buffer);
    layer_factory->setInputTensor(down_proj_weight.get_tensor(), 22);

    fp = nullptr;
    r = fopen_s(&fp, ("D:/kai/remote-tensor/qwen-dumps/model_weights/model_"+std::to_string(idx)+"_input_23.bin").c_str(), "rb");
    fseek(fp, 0, SEEK_END);
    nSize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    half_ptr down_proj_scale_buffer = new uint16_t[148 * 3584];
    fread(down_proj_scale_buffer, 2, nSize/2, fp);
    auto down_proj_scale = Tensor(ov::element::Type_t::f16, ov::Shape({148, 1, 3584}), down_proj_scale_buffer);
    layer_factory->setInputTensor(down_proj_scale.get_tensor(), 23);

    layer_factory->setOutputTensor(hidden_buffer, 0);
    half_ptr k_result = new uint16_t[1 * 4 * 128];
    half_ptr v_result = new uint16_t[1 * 4 * 128];
    memset(k_result, 0, 4 * 128 * sizeof(uint16_t));
    memset(v_result, 0, 4 * 128 * sizeof(uint16_t));
    layer_factory->setOutputTensor(k_result, 1);
    layer_factory->setOutputTensor(v_result, 2);

    embedding_factory->run();
    layer_factory->run();
    half_ptr hidden_output = (half_ptr)hidden_buffer.get();
    std::cout << unsigned(hidden_output[0]) << " " << unsigned(hidden_output[1]) << std::endl;
    std::cout << unsigned(k_result[0]) << " " << unsigned(k_result[100]) << std::endl;
    std::cout << unsigned(v_result[0]) << " " << unsigned(v_result[100]) << std::endl;
    system("pause");
    return 0;
}