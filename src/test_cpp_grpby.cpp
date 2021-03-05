#include <iostream>
#include <memory>
#include <vector>
#include <algorithm>
#include <random>
#include <utility>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/groupby.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/copying.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/filling.hpp>
#include <cudf/concatenate.hpp>

#include <cudf_test/column_utilities.hpp>

std::unique_ptr<cudf::table> make_key_table(const int& num_row, const int& num_group) {
    int group_size = num_row / num_group;
    auto single_group = cudf::sequence(cudf::data_type(cudf::data_id(INT32)), group_size);
    auto key = cudf::repeat(cudf::table{single_group)}, num_group);
    return key;
}

std::unique_ptr<cudf::table> make_val_table(const int& num_rows, const int& num_cols) {
    std::vector<std::unique_ptr<cudf::column>> cols;

    for (int i = 0; i < num_cols; i++) {
        auto scalar = cudf::make_fixed_width_scalar((i + 1) * 2);
        cols.push_back(std::move(cudf::make_column_from_scalar(*scalar, num_rows)));
    }

    return std::make_unique<cudf::table>(std::move(cols));
}

std::unique_ptr<cudf::table> grpby_elementwise_mul(cudf::table_view const& key, cudf::table_view const& cols) {
    cudf::groupby::groupby grpby_obj(key);
    cudf::groupby::groupby::groups groups = grpby_obj.get_groups(cudf::table_view(cols));

    std::vector<std::unique_ptr<cudf::column>> results;
    std::vector<std::vector<std::unique_ptr<cudf::column>>> slice_results;

    for (int i = 0; i < cols.num_columns(); i++) {
        nvtx3::thread_range r1{"Column loop"};
        auto col_view = groups.values->get_column(i).view();
        slice_results.push_back(std::vector<std::unique_ptr<cudf::column>>());
        auto &slice_result = slice_results[i];
        
        std::vector<cudf::size_type> slice_indice;
        slice_indice.resize((groups.offsets.size() -1) * 2);
        for (int i = 1; i < groups.offsets.size()-1; i++) {
            slice_indice[2 * i - 1] = groups.offsets[i];
            slice_indice[2 * i] = groups.offsets[i];
        }
        *slice_indice.begin() = *groups.offsets.begin();
        slice_indice.back() = groups.offsets.back();

        auto group_values = cudf::slice(col_view, slice_indice);
        // To benchmark: Compute the slices and store result
        for (auto val : group_values) {
            auto res = cudf::binary_operation(val, val, cudf::binary_operator::MUL, val.type());
            slice_result.push_back(std::move(res));
        }
    }

    // Concat
    std::for_each(slice_results.begin(), slice_results.end(), [&results](auto const& slices){
        std::vector<cudf::column_view> views;
        std::for_each(slices.begin(), slices.end(), [&views](auto const& slice){
            views.push_back(slice->view());
        });
        results.push_back(std::move(cudf::concatenate(views)));
    });

    return std::make_unique<cudf::table>(std::move(results));
}

int main(int argc, char** argv) {
    int num_rows, num_groups, num_cols;
    num_rows = std::stoi(argv[1]);
    num_groups = std::stoi(argv[2]);
    num_cols = std::stoi(argv[3]);

    size_t available_mem, total_mem, prealloc_mem;
    cudaMemGetInfo(&available_mem, &total_mem);
    prealloc_mem = total_mem * 0.85;
    prealloc_mem = prealloc_mem - prealloc_mem % 256;
    
    auto cuda_mr = std::make_unique<rmm::mr::cuda_memory_resource>();
    auto pool_mr = std::make_unique<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>>(cuda_mr.get(), prealloc_mem);
    rmm::mr::set_current_device_resource(pool_mr.get());

    auto key = make_key_table(num_rows, num_groups);
    auto values = make_val_table(num_rows, num_cols);

    // cudf::test::print(key->view(), std::cout, ", ");
    // for (int i = 0; i < values->num_columns(); i++) {
    //     std::cout << "Input Column: " << i << ": ";
    //     cudf::test::print(values->get_column(i).view(), std::cout, ", ");
    // }

    auto res = grpby_elementwise_mul(key->view(), values->view());
    // for (int i = 0; i < res->num_columns(); i++) {
    //     auto col = res->get_column(i);
    //     std::cout << "Output Column: " << i << ": ";
    //     cudf::test::print(col.view(), std::cout, ", ");
    // }

}
