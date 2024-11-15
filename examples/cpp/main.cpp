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
    const size_t batch = 1, inC = 4096, outC = 150000, N = 200;
    // const size_t batch = 128, inC = 256, outC = 512, N = 10000;

    std::cout << "Create a ModelFactory" << std::endl;
    auto factory = std::make_shared<ModelFactory>("NPU");

    // create parameter
    auto input = factory->parameter({batch, inC}, ov::element::f16);
    auto weights = factory->parameter({outC, inC}, ov::element::f16);
    auto bias = factory->parameter({1, outC}, ov::element::f16);

    // create matmul
    auto matmul = factory->matmul(input, weights);
    auto matmul_bias = factory->eltwise_add(matmul, bias);
    factory->result(matmul_bias);

    // Compile the model
    factory->compile();
    // system("pause");

    // Save OV model
    // std::cout << "Saving model to matmul.xml" << std::endl;
    // factory->saveModel("matmul.xml");

    // std::cout << "Creating tensor buffers" << std::endl;
    // half_ptr input_buffer = new uint16_t[batch * inC];
    // half_ptr weights_buffer = new uint16_t[outC * inC];
    // half_ptr bias_buffer = new uint16_t[outC];
    // half_ptr output_buffer = new uint16_t[batch * outC];

    // memset(input_buffer, 0, batch * inC * sizeof(uint16_t));
    // memset(weights_buffer, 0, outC * inC * sizeof(uint16_t));
    // memset(bias_buffer, 0, outC * sizeof(uint16_t));
    // memset(output_buffer, 0, batch * outC * sizeof(uint16_t));

    std::cout << "Creating a remote tensor" << std::endl;
    auto input_buffer = factory->createRemoteInputTensor(0);
    auto weights_buffer = factory->createRemoteInputTensor(1);
    auto bias_buffer = factory->createRemoteInputTensor(2);
    auto output_buffer = factory->createRemoteOutputTensor(0);
    // system("pause");

    std::memset(input_buffer.get(), 0, input_buffer.get_byte_size());
    std::memset(weights_buffer.get(), 0, weights_buffer.get_byte_size());
    std::memset(bias_buffer.get(), 0, bias_buffer.get_byte_size());
    std::memset(output_buffer.get(), 0, output_buffer.get_byte_size());
    // system("pause");

    factory->setInputTensor(input_buffer, 0);
    factory->setInputTensor(weights_buffer, 1);
    factory->setInputTensor(bias_buffer, 2);
    factory->setOutputTensor(output_buffer, 0);
    // system("pause");

    // Run inference
    std::cout << "Run inference on " << N << " workloads" << std::endl;
    auto start = high_resolution_clock::now();
    for (auto idx = 0; idx < N; idx++) {
        factory->run();
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    std::cout << "Average time used " << (double)duration.count()/N << std::endl;
    system("pause");

    std::cout << "Inference done" << std::endl;
    return 0;
}