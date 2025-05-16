#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <typeindex>

class FlexParam {
public:
    ~FlexParam();

    void set(const std::string& name, const void* data, size_t len, std::type_index type);

    // 單一接口：回傳 void* 和 size
    void* get(const std::string& name, size_t& size) const;

    // 額外資訊查詢
    std::type_index getType(const std::string& name) const;

    void printAll() const;

private:
    std::vector<void*> data_;
    std::vector<std::type_index> types_;
    std::vector<size_t> lengths_;
    std::unordered_map<std::string, size_t> name_to_index_;
};
