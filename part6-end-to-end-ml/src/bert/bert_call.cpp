// Call the compiled bert-base-uncased from C++ through the _mlir_ciface_
// wrapper. Two things are new compared to the single-tensor models:
//
//  1. The model has TWO lifted buffers as its first two arguments
//     (embeddings.position_ids, embeddings.token_type_ids — torch-mlir does
//     not freeze buffers, Chapter 27 section 7). get_buffers.py dumps them to
//     tensor.csv; load_tensor() reads them back here.
//  2. The model returns TWO tensors. --llvm-request-c-wrappers then passes a
//     single out-pointer to a struct holding one descriptor per result, in
//     result order — the Output struct below.
//
// Build: python get_buffers.py && sh run_mlir_pipeline.sh && ./bert
// (see Chapter 30). Prints the same numbers as run_bert_model.py.
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// Load the buffers saved from PyTorch by get_buffers.py: one CSV row each.
std::vector<std::vector<int64_t>> load_tensor(const char *path) {
  std::vector<std::vector<int64_t>> result;
  std::ifstream file(path);
  if (!file) {
    std::cerr << "cannot open " << path << " (run get_buffers.py first)\n";
    std::exit(1);
  }
  std::string line, token;
  while (std::getline(file, line)) {
    std::stringstream ss(line);
    std::vector<int64_t> row;
    while (std::getline(ss, token, ','))
      row.push_back(std::stoll(token));
    result.push_back(row);
  }
  return result;
}

template <typename T, int N> struct MemRefDescriptor {
  T *allocated;
  T *aligned;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};

// One descriptor per function result, in order.
struct Output {
  MemRefDescriptor<float, 3> last_hidden_state; // [1, 12, 768]
  MemRefDescriptor<float, 2> pooler_output;     // [1, 768]
};

extern "C" {
void _mlir_ciface_bert_model(
    Output *result,
    MemRefDescriptor<int64_t, 2> *buffer_embeddings_position_ids,
    MemRefDescriptor<int64_t, 2> *buffer_embeddings_token_type_ids,
    MemRefDescriptor<int64_t, 2> *input_ids,
    MemRefDescriptor<int64_t, 2> *token_type_ids,
    MemRefDescriptor<int64_t, 2> *attention_mask);
}

int main() {
  auto buffers = load_tensor("tensor.csv");
  if (buffers.size() != 2 || buffers[0].size() != 512 ||
      buffers[1].size() != 512) {
    std::cerr << "tensor.csv: expected 2 rows of 512 values\n";
    return 1;
  }

  // "This is a sample input for BERT model export." (BertTokenizer)
  int64_t input_ids[1][12] = {
      {101, 2023, 2003, 1037, 7099, 7953, 2005, 14324, 2944, 9167, 1012, 102}};
  int64_t token_type_ids[1][12] = {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
  int64_t attention_mask[1][12] = {{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}};

  MemRefDescriptor<int64_t, 2> position_ids_buf = {
      buffers[0].data(), buffers[0].data(), 0, {1, 512}, {512, 1}};
  MemRefDescriptor<int64_t, 2> token_type_ids_buf = {
      buffers[1].data(), buffers[1].data(), 0, {1, 512}, {512, 1}};

  MemRefDescriptor<int64_t, 2> input_ids_MemRef = {
      (int64_t *)input_ids, (int64_t *)input_ids, 0, {1, 12}, {12, 1}};
  MemRefDescriptor<int64_t, 2> token_type_ids_MemRef = {
      (int64_t *)token_type_ids, (int64_t *)token_type_ids, 0, {1, 12},
      {12, 1}};
  MemRefDescriptor<int64_t, 2> attention_mask_MemRef = {
      (int64_t *)attention_mask, (int64_t *)attention_mask, 0, {1, 12},
      {12, 1}};

  // Out-slot: both descriptors are overwritten with model-allocated
  // buffers (Chapter 28, section 3).
  Output result = {};

  _mlir_ciface_bert_model(&result, &position_ids_buf, &token_type_ids_buf,
                          &input_ids_MemRef, &token_type_ids_MemRef,
                          &attention_mask_MemRef);

  const MemRefDescriptor<float, 3> &h = result.last_hidden_state;
  const MemRefDescriptor<float, 2> &p = result.pooler_output;
  const float *hidden = h.aligned + h.offset;
  const float *pooled = p.aligned + p.offset;

  std::cout << "Last hidden state (first 10 values of each of the "
            << h.sizes[1] << " vectors):\n";
  for (int64_t i = 0; i < h.sizes[1]; ++i) {
    for (int64_t j = 0; j < 10; ++j)
      std::cout << std::fixed << std::setprecision(5)
                << hidden[i * h.strides[1] + j * h.strides[2]] << ' ';
    std::cout << '\n';
  }
  std::cout << "\nPooler output:\n";
  for (int64_t j = 0; j < 10; ++j)
    std::cout << std::fixed << std::setprecision(5)
              << pooled[j * p.strides[1]] << ' ';
  std::cout << '\n';

  free(h.allocated);
  free(p.allocated);
  return 0;
}
