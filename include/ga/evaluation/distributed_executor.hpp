#pragma once

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <future>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace ga {
namespace evaluation {

// Extension point for remote/distributed fitness evaluation backends.
class IDistributedExecutor {
public:
    virtual ~IDistributedExecutor() = default;
    virtual std::vector<double> execute(const std::vector<std::vector<double>>& batch) = 0;
};

class LocalDistributedExecutor : public IDistributedExecutor {
public:
    using EvaluateFn = std::function<double(const std::vector<double>&)>;

    explicit LocalDistributedExecutor(EvaluateFn evaluator,
                                      std::size_t workers = std::thread::hardware_concurrency())
        : evaluator_(std::move(evaluator)), workers_(std::max<std::size_t>(1, workers)) {
        if (!evaluator_) {
            throw std::invalid_argument("LocalDistributedExecutor evaluator is empty");
        }
    }

    std::vector<double> execute(const std::vector<std::vector<double>>& batch) override {
        std::vector<double> out(batch.size());
        if (batch.empty()) {
            return out;
        }

        const std::size_t chunk = (batch.size() + workers_ - 1) / workers_;
        std::vector<std::future<void>> tasks;
        tasks.reserve(workers_);

        for (std::size_t w = 0; w < workers_; ++w) {
            const std::size_t begin = w * chunk;
            if (begin >= batch.size()) {
                break;
            }
            const std::size_t end = std::min(batch.size(), begin + chunk);
            tasks.push_back(std::async(std::launch::async, [&, begin, end]() {
                for (std::size_t i = begin; i < end; ++i) {
                    out[i] = evaluator_(batch[i]);
                }
            }));
        }

        for (auto& t : tasks) {
            t.get();
        }
        return out;
    }

private:
    EvaluateFn evaluator_;
    std::size_t workers_;
};

class StubDistributedExecutor : public IDistributedExecutor {
public:
    std::vector<double> execute(const std::vector<std::vector<double>>& batch) override {
        throw std::runtime_error("Distributed execution backend is not configured. Batch size=" +
                                 std::to_string(batch.size()));
    }
};

#if defined(__unix__) || defined(__APPLE__)

class ProcessDistributedExecutor : public IDistributedExecutor {
public:
    using EvaluateFn = std::function<double(const std::vector<double>&)>;

    explicit ProcessDistributedExecutor(EvaluateFn evaluator,
                                        std::size_t workers = std::thread::hardware_concurrency())
        : evaluator_(std::move(evaluator)), workers_(std::max<std::size_t>(1, workers)) {
        if (!evaluator_) {
            throw std::invalid_argument("ProcessDistributedExecutor evaluator is empty");
        }
    }

    std::vector<double> execute(const std::vector<std::vector<double>>& batch) override {
        std::vector<double> out(batch.size());
        if (batch.empty()) {
            return out;
        }

        const std::size_t workerCount = std::min(workers_, batch.size());
        std::vector<Worker> workers;
        workers.reserve(workerCount);

        try {
            for (std::size_t i = 0; i < workerCount; ++i) {
                workers.push_back(spawnWorker());
            }

            for (std::size_t i = 0; i < batch.size(); ++i) {
                Worker& w = workers[i % workerCount];
                TaskHeader header;
                header.index = static_cast<std::int64_t>(i);
                header.length = static_cast<std::uint64_t>(batch[i].size());

                if (!writeExact(w.parentToChildWriteFd, &header, sizeof(header))) {
                    throw std::runtime_error("ProcessDistributedExecutor failed sending task header");
                }
                if (!batch[i].empty() &&
                    !writeExact(w.parentToChildWriteFd,
                                batch[i].data(),
                                sizeof(double) * batch[i].size())) {
                    throw std::runtime_error("ProcessDistributedExecutor failed sending task payload");
                }
                w.assigned += 1;
            }

            TaskHeader done;
            done.index = -1;
            done.length = 0;
            for (auto& w : workers) {
                if (!writeExact(w.parentToChildWriteFd, &done, sizeof(done))) {
                    throw std::runtime_error("ProcessDistributedExecutor failed sending shutdown signal");
                }
                closeFd(w.parentToChildWriteFd);
            }

            for (auto& w : workers) {
                for (std::size_t c = 0; c < w.assigned; ++c) {
                    ResultMsg msg;
                    if (!readExact(w.childToParentReadFd, &msg, sizeof(msg))) {
                        throw std::runtime_error("ProcessDistributedExecutor failed reading worker result");
                    }
                    if (msg.index < 0 || static_cast<std::size_t>(msg.index) >= out.size()) {
                        throw std::runtime_error("ProcessDistributedExecutor received invalid result index");
                    }
                    out[static_cast<std::size_t>(msg.index)] = msg.value;
                }
                closeFd(w.childToParentReadFd);
            }

            for (auto& w : workers) {
                int status = 0;
                if (waitpid(w.pid, &status, 0) < 0) {
                    throw std::runtime_error("ProcessDistributedExecutor waitpid failed: " +
                                             std::string(std::strerror(errno)));
                }
                if (!(WIFEXITED(status) && WEXITSTATUS(status) == 0)) {
                    throw std::runtime_error("ProcessDistributedExecutor worker exited abnormally");
                }
            }
        } catch (...) {
            cleanupWorkers(workers);
            throw;
        }

        return out;
    }

private:
    struct TaskHeader {
        std::int64_t index = 0;
        std::uint64_t length = 0;
    };

    struct ResultMsg {
        std::int64_t index = 0;
        double value = 0.0;
    };

    struct Worker {
        pid_t pid = -1;
        int parentToChildWriteFd = -1;
        int childToParentReadFd = -1;
        std::size_t assigned = 0;
    };

    static void closeFd(int& fd) {
        if (fd >= 0) {
            (void)close(fd);
            fd = -1;
        }
    }

    static bool writeExact(int fd, const void* data, std::size_t bytes) {
        const auto* ptr = static_cast<const std::uint8_t*>(data);
        std::size_t remaining = bytes;
        while (remaining > 0) {
            const ssize_t n = write(fd, ptr, remaining);
            if (n < 0) {
                if (errno == EINTR) {
                    continue;
                }
                return false;
            }
            if (n == 0) {
                return false;
            }
            ptr += static_cast<std::size_t>(n);
            remaining -= static_cast<std::size_t>(n);
        }
        return true;
    }

    static bool readExact(int fd, void* data, std::size_t bytes) {
        auto* ptr = static_cast<std::uint8_t*>(data);
        std::size_t remaining = bytes;
        while (remaining > 0) {
            const ssize_t n = read(fd, ptr, remaining);
            if (n < 0) {
                if (errno == EINTR) {
                    continue;
                }
                return false;
            }
            if (n == 0) {
                return false;
            }
            ptr += static_cast<std::size_t>(n);
            remaining -= static_cast<std::size_t>(n);
        }
        return true;
    }

    Worker spawnWorker() const {
        int toChild[2] = {-1, -1};
        int fromChild[2] = {-1, -1};
        if (pipe(toChild) != 0 || pipe(fromChild) != 0) {
            closeFd(toChild[0]);
            closeFd(toChild[1]);
            closeFd(fromChild[0]);
            closeFd(fromChild[1]);
            throw std::runtime_error("ProcessDistributedExecutor pipe creation failed: " +
                                     std::string(std::strerror(errno)));
        }

        const pid_t pid = fork();
        if (pid < 0) {
            closeFd(toChild[0]);
            closeFd(toChild[1]);
            closeFd(fromChild[0]);
            closeFd(fromChild[1]);
            throw std::runtime_error("ProcessDistributedExecutor fork failed: " +
                                     std::string(std::strerror(errno)));
        }

        if (pid == 0) {
            (void)close(toChild[1]);
            (void)close(fromChild[0]);

            for (;;) {
                TaskHeader header;
                if (!readExact(toChild[0], &header, sizeof(header))) {
                    break;
                }
                if (header.index < 0) {
                    break;
                }

                std::vector<double> genes(static_cast<std::size_t>(header.length));
                if (!genes.empty() &&
                    !readExact(toChild[0], genes.data(), sizeof(double) * genes.size())) {
                    break;
                }

                ResultMsg msg;
                msg.index = header.index;
                msg.value = evaluator_(genes);
                if (!writeExact(fromChild[1], &msg, sizeof(msg))) {
                    break;
                }
            }

            (void)close(toChild[0]);
            (void)close(fromChild[1]);
            _exit(0);
        }

        closeFd(toChild[0]);
        closeFd(fromChild[1]);

        Worker w;
        w.pid = pid;
        w.parentToChildWriteFd = toChild[1];
        w.childToParentReadFd = fromChild[0];
        return w;
    }

    static void cleanupWorkers(std::vector<Worker>& workers) {
        for (auto& w : workers) {
            if (w.parentToChildWriteFd >= 0) {
                TaskHeader done;
                done.index = -1;
                done.length = 0;
                (void)writeExact(w.parentToChildWriteFd, &done, sizeof(done));
            }
            closeFd(w.parentToChildWriteFd);
            closeFd(w.childToParentReadFd);
        }

        for (auto& w : workers) {
            if (w.pid > 0) {
                int status = 0;
                (void)waitpid(w.pid, &status, 0);
            }
        }
    }

    EvaluateFn evaluator_;
    std::size_t workers_;
};

#else

class ProcessDistributedExecutor : public IDistributedExecutor {
public:
    using EvaluateFn = std::function<double(const std::vector<double>&)>;

    explicit ProcessDistributedExecutor(EvaluateFn /*evaluator*/,
                                        std::size_t /*workers*/ = 1) {}

    std::vector<double> execute(const std::vector<std::vector<double>>& /*batch*/) override {
        throw std::runtime_error("ProcessDistributedExecutor is only supported on POSIX platforms");
    }
};

#endif

} // namespace evaluation
} // namespace ga
