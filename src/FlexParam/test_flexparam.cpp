#include "FlexParam.hpp"
#include <iostream>

int main() {
    FlexParam param;

    int arr[] = {10, 20, 30};
    float val = 3.14f;

    param.set("ints", arr, sizeof(arr), typeid(int));
    param.set("floatVal", &val, sizeof(val), typeid(float));

    size_t size;
    void* raw_ptr = param.get("ints", size);
    int* int_ptr = static_cast<int*>(raw_ptr);
    size_t count = size / sizeof(int);

    for (size_t i = 0; i < count; ++i) {
        std::cout << "ints[" << i << "] = " << int_ptr[i] << "\n";
    }

    raw_ptr = param.get("floatVal", size);
    std::cout << "floatVal = " << *static_cast<float*>(raw_ptr) << std::endl;

    return 0;
}
