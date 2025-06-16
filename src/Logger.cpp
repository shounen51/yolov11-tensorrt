#include "Logger.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <mutex>
#include <chrono>
#include <thread>
#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif
#include <filesystem>

class YoloLogger::Impl {
public:
    Impl(const std::string& logFilePath)
        : logFilePath_(logFilePath), maxFileSize_(10 * 1024 * 1024) // 10MB
    {
        std::filesystem::path logPath(logFilePath_);
        if (logPath.has_parent_path()) {
            std::error_code ec;
            std::filesystem::create_directories(logPath.parent_path(), ec);
        }
        openLogFile();
    }
    ~Impl() { logFile.close(); }

    void log(LogLevel level, const std::string& file, int line, const std::string& message) {
        std::lock_guard<std::mutex> lock(mtx);
        rotateIfNeeded();
        std::string logStr = formatLog(level, file, line, message);
        logFile << logStr << std::endl;
        std::cout << logStr << std::endl; // 同時印出到 console
    }

private:
    std::ofstream logFile;
    std::mutex mtx;
    std::string logFilePath_;
    std::uintmax_t maxFileSize_;

    void openLogFile() {
        logFile.open(logFilePath_, std::ios::app);
    }

    void rotateIfNeeded() {
        logFile.flush();
        std::error_code ec;
        auto size = std::filesystem::file_size(logFilePath_, ec);
        if (!ec && size >= maxFileSize_) {
            logFile.close();
            std::string rotated = logFilePath_ + ".old";
            std::filesystem::remove(rotated, ec); // ignore error
            std::filesystem::rename(logFilePath_, rotated, ec); // ignore error
            openLogFile();
        }
    }

    std::string levelToString(LogLevel level) {
        switch (level) {
            case LogLevel::DEBUG: return "DEBUG";
            case LogLevel::INFO:  return "INFO";
            case LogLevel::WARN:  return "WARN";
            case LogLevel::ERROR: return "ERROR";
            default: return "UNKNOWN";
        }
    }

    std::string formatLog(LogLevel level, const std::string& file, int line, const std::string& message) {
        std::ostringstream oss;
        // Timestamp
        auto now = std::chrono::system_clock::now();
        std::time_t t = std::chrono::system_clock::to_time_t(now);
        std::tm tm;
    #if defined(_WIN32)
        localtime_s(&tm, &t);
    #else
        localtime_r(&t, &tm);
    #endif
        oss << "[" << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") << "]";
        // PID
    #if defined(_WIN32)
        oss << " - [PID " << _getpid() << "]";
    #else
        oss << " - [PID " << getpid() << "]";
    #endif
        // Level
        oss << " - [" << levelToString(level) << "]";
        // File:Line
        std::string filename = file.substr(file.find_last_of("/\\") + 1);
        oss << " [" << filename << ":" << line << "]";
        // Message
        oss << " : " << message;
        return oss.str();
    }
};

static std::unique_ptr<YoloLogger> g_logger;

void YoloLogger::init(const std::string& logFilePath) {
    g_logger.reset(new YoloLogger(logFilePath));
}

YoloLogger& YoloLogger::instance() {
    if (!g_logger) {
        throw std::runtime_error("YoloLogger not initialized. Call YoloLogger::init() first.");
    }
    return *g_logger;
}

YoloLogger::YoloLogger(const std::string& logFilePath) : pImpl(new Impl(logFilePath)) {}
YoloLogger::~YoloLogger() = default;

void YoloLogger::log(LogLevel level, const std::string& file, int line, const std::string& message) {
    pImpl->log(level, file, line, message);
}
