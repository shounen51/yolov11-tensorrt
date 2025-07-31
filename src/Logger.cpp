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

static bool g_consoleOnly = false;

void AILogger::setConsoleOnly(bool enable) {
    g_consoleOnly = enable;
}

class AILogger::Impl {
public:
    Impl(const std::string& logFilePath)
        : logFilePath_(logFilePath), maxFileSize_(512 * 1024 * 1024) // 512MB
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
        std::string logStr = formatLog(level, file, line, message);
        if (g_consoleOnly) {
            // 只輸出到 console
            std::cout << logStr << std::endl;
        } else {
            // 只寫入檔案，不輸出到 console
            rotateIfNeeded();
            logFile << logStr << std::endl;
        }
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

            const int maxLogFiles = 200; // 可調整的保留檔案數量

            // 只刪除最舊的檔案，然後將當前檔案移動到 .1
            std::string oldestFile = logFilePath_ + "." + std::to_string(maxLogFiles);
            std::filesystem::remove(oldestFile, ec); // 刪除最舊的檔案

            // 將當前檔案重命名為帶時間戳的檔案名，避免大量重命名操作
            auto now = std::chrono::system_clock::now();
            std::time_t t = std::chrono::system_clock::to_time_t(now);
            std::tm tm;
        #if defined(_WIN32)
            localtime_s(&tm, &t);
        #else
            localtime_r(&t, &tm);
        #endif

            std::ostringstream oss;
            oss << logFilePath_ << "." << std::put_time(&tm, "%Y%m%d_%H%M%S");
            std::string timestampedFile = oss.str();

            std::filesystem::rename(logFilePath_, timestampedFile, ec);

            // 清理舊檔案：保持檔案數量不超過 maxLogFiles
            cleanupOldLogFiles(maxLogFiles);

            openLogFile();
        }
    }

    void cleanupOldLogFiles(int maxFiles) {
        std::error_code ec;
        std::filesystem::path logDir = std::filesystem::path(logFilePath_).parent_path();
        std::string logBaseName = std::filesystem::path(logFilePath_).filename().string();

        std::vector<std::filesystem::directory_entry> logFiles;

        // 收集所有相關的log檔案
        for (const auto& entry : std::filesystem::directory_iterator(logDir, ec)) {
            if (entry.is_regular_file() &&
                entry.path().filename().string().find(logBaseName + ".") == 0) {
                logFiles.push_back(entry);
            }
        }

        // 按修改時間排序（最新的在前）
        std::sort(logFiles.begin(), logFiles.end(),
                  [](const auto& a, const auto& b) {
                      std::error_code ec;
                      return std::filesystem::last_write_time(a, ec) >
                             std::filesystem::last_write_time(b, ec);
                  });

        // 刪除超過限制的舊檔案
        for (size_t i = maxFiles; i < logFiles.size(); ++i) {
            std::filesystem::remove(logFiles[i].path(), ec);
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
        // Timestamp with milliseconds
        auto now = std::chrono::system_clock::now();
        std::time_t t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
        std::tm tm;
    #if defined(_WIN32)
        localtime_s(&tm, &t);
    #else
        localtime_r(&t, &tm);
    #endif
        oss << "[" << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") << "." << std::setfill('0') << std::setw(3) << ms.count() << "]";
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

static std::unique_ptr<AILogger> g_logger;

void AILogger::init(const std::string& logFilePath) {
    g_logger.reset(new AILogger(logFilePath));
}

AILogger& AILogger::instance() {
    if (!g_logger) {
        throw std::runtime_error("AILogger not initialized. Call AILogger::init() first.");
    }
    return *g_logger;
}

AILogger::AILogger(const std::string& logFilePath) : pImpl(new Impl(logFilePath)) {}
AILogger::~AILogger() = default;

void AILogger::log(LogLevel level, const std::string& file, int line, const std::string& message) {
    pImpl->log(level, file, line, message);
}
