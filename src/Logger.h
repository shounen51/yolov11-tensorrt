#pragma once
#include <string>
#include <memory>

// Logger configuration constants
namespace LoggerConfig {
    constexpr std::uintmax_t MAX_LOG_FILE_SIZE = 512 * 1024 * 1024;  // 512MB
    constexpr int MAX_LOG_FILES = 200;  // Maximum number of log files to keep
}

enum class LogLevel { DEBUG, INFO, WARN, ERROR };

class AILogger {
public:
    static void init(const std::string& logFilePath);
    static AILogger& instance();

    // 設定只輸出到 console
    static void setConsoleOnly(bool enable);

    void log(LogLevel level, const std::string& file, int line, const std::string& message);

    // Convenience macros for logging with file and line info
    #define AILOG_DEBUG(msg) AILogger::instance().log(LogLevel::DEBUG, __FILE__, __LINE__, msg)
    #define AILOG_INFO(msg)  AILogger::instance().log(LogLevel::INFO,  __FILE__, __LINE__, msg)
    #define AILOG_WARN(msg)  AILogger::instance().log(LogLevel::WARN,  __FILE__, __LINE__, msg)
    #define AILOG_ERROR(msg) AILogger::instance().log(LogLevel::ERROR, __FILE__, __LINE__, msg)

    ~AILogger();

private:
    AILogger(const std::string& logFilePath);
    AILogger(const AILogger&) = delete;
    AILogger& operator=(const AILogger&) = delete;

    class Impl;
    std::unique_ptr<Impl> pImpl;
};
