#include "FlexParam.h"
#include <iostream>
#include <cstring>

FlexParam::~FlexParam() {
    for (void* ptr : data_) {
        delete[] reinterpret_cast<char*>(ptr);
    }
}

void FlexParam::set(const std::string& name, const void* data, size_t len, std::type_index type) {
    auto it = name_to_index_.find(name);

    void* new_data = new char[len];
    std::memcpy(new_data, data, len);

    if (it != name_to_index_.end()) {
        size_t idx = it->second;
        delete[] reinterpret_cast<char*>(data_[idx]);
        data_[idx] = new_data;
        types_[idx] = type;
        lengths_[idx] = len;
    } else {
        size_t idx = data_.size();
        name_to_index_[name] = idx;
        data_.push_back(new_data);
        types_.push_back(type);
        lengths_.push_back(len);
    }
}

void* FlexParam::get(const std::string& name, size_t& size) const {
    auto it = name_to_index_.find(name);
    if (it == name_to_index_.end()) {
        throw std::runtime_error("Name not found: " + name);
    }
    size_t idx = it->second;
    size = lengths_[idx];
    return data_[idx];
}

std::type_index FlexParam::getType(const std::string& name) const {
    auto it = name_to_index_.find(name);
    if (it == name_to_index_.end()) {
        throw std::runtime_error("Name not found: " + name);
    }
    return types_[it->second];
}

void FlexParam::printAll() const {
    for (const auto& [name, idx] : name_to_index_) {
        std::cout << "Name: " << name
                  << ", Type: " << types_[idx].name()
                  << ", Length: " << lengths_[idx] << " bytes"
                  << std::endl;
    }
}
