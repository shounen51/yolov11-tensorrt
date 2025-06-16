#pragma once
#include <string>
#include <memory>

enum class LogLevel { DEBUG, INFO, WARN, ERROR };

class YoloLogger {
public:
    static void init(const std::string& logFilePath);
    static YoloLogger& instance();

    void log(LogLevel level, const std::string& file, int line, const std::string& message);

    // Convenience macros for logging with file and line info
    #define LOG_DEBUG(msg) YoloLogger::instance().log(LogLevel::DEBUG, __FILE__, __LINE__, msg)
    #define LOG_INFO(msg)  YoloLogger::instance().log(LogLevel::INFO,  __FILE__, __LINE__, msg)
    #define LOG_WARN(msg)  YoloLogger::instance().log(LogLevel::WARN,  __FILE__, __LINE__, msg)
    #define LOG_ERROR(msg) YoloLogger::instance().log(LogLevel::ERROR, __FILE__, __LINE__, msg)

    ~YoloLogger(); // <-- 移到 public

private:
    YoloLogger(const std::string& logFilePath);
    YoloLogger(const YoloLogger&) = delete;
    YoloLogger& operator=(const YoloLogger&) = delete;

    class Impl;
    std::unique_ptr<Impl> pImpl;
};
