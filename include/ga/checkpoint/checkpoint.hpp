#pragma once

#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ga/config.hpp"

namespace ga {
namespace checkpoint {

struct CheckpointState {
    ga::Config config;
    ga::Result result;
    int generation = 0;
    std::string rngState;
};

class CheckpointManager {
public:
    static void saveBinary(const std::string& path, const CheckpointState& state) {
        std::ofstream out(path, std::ios::binary);
        if (!out) {
            throw std::runtime_error("Unable to open checkpoint file for writing: " + path);
        }

        writePod(out, state.config.populationSize);
        writePod(out, state.config.generations);
        writePod(out, state.config.dimension);
        writePod(out, state.config.crossoverRate);
        writePod(out, state.config.mutationRate);
        writePod(out, state.config.bounds.lower);
        writePod(out, state.config.bounds.upper);
        writePod(out, state.config.eliteRatio);
        writePod(out, state.config.seed);

        writePod(out, state.generation);
        writeString(out, state.rngState);
        writeVector(out, state.result.bestGenes);
        writePod(out, state.result.bestFitness);
        writeVector(out, state.result.bestHistory);
        writeVector(out, state.result.avgHistory);
    }

    static CheckpointState loadBinary(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            throw std::runtime_error("Unable to open checkpoint file for reading: " + path);
        }

        CheckpointState state;
        readPod(in, state.config.populationSize);
        readPod(in, state.config.generations);
        readPod(in, state.config.dimension);
        readPod(in, state.config.crossoverRate);
        readPod(in, state.config.mutationRate);
        readPod(in, state.config.bounds.lower);
        readPod(in, state.config.bounds.upper);
        readPod(in, state.config.eliteRatio);
        readPod(in, state.config.seed);

        readPod(in, state.generation);
        state.rngState = readString(in);
        state.result.bestGenes = readVector(in);
        readPod(in, state.result.bestFitness);
        state.result.bestHistory = readVector(in);
        state.result.avgHistory = readVector(in);

        return state;
    }

    static void saveJson(const std::string& path, const CheckpointState& state) {
        std::ofstream out(path);
        if (!out) {
            throw std::runtime_error("Unable to open checkpoint JSON file for writing: " + path);
        }

        out << std::setprecision(std::numeric_limits<double>::max_digits10);
        out << "{\n";
        out << "  \"config\": {\n";
        out << "    \"population_size\": " << state.config.populationSize << ",\n";
        out << "    \"generations\": " << state.config.generations << ",\n";
        out << "    \"dimension\": " << state.config.dimension << ",\n";
        out << "    \"crossover_rate\": " << state.config.crossoverRate << ",\n";
        out << "    \"mutation_rate\": " << state.config.mutationRate << ",\n";
        out << "    \"bounds\": {\n";
        out << "      \"lower\": " << state.config.bounds.lower << ",\n";
        out << "      \"upper\": " << state.config.bounds.upper << "\n";
        out << "    },\n";
        out << "    \"elite_ratio\": " << state.config.eliteRatio << ",\n";
        out << "    \"seed\": " << state.config.seed << "\n";
        out << "  },\n";
        out << "  \"generation\": " << state.generation << ",\n";
        out << "  \"rng_state\": \"" << escapeJsonString(state.rngState) << "\",\n";
        out << "  \"result\": {\n";
        out << "    \"best_genes\": " << vectorToJson(state.result.bestGenes) << ",\n";
        out << "    \"best_fitness\": " << state.result.bestFitness << ",\n";
        out << "    \"best_history\": " << vectorToJson(state.result.bestHistory) << ",\n";
        out << "    \"avg_history\": " << vectorToJson(state.result.avgHistory) << "\n";
        out << "  }\n";
        out << "}\n";
    }

    static CheckpointState loadJson(const std::string& path) {
        std::ifstream in(path);
        if (!in) {
            throw std::runtime_error("Unable to open checkpoint JSON file for reading: " + path);
        }

        const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        if (text.empty()) {
            throw std::runtime_error("Checkpoint JSON file is empty: " + path);
        }

        CheckpointState state;
        state.config.populationSize = parseIntKey(text, "population_size");
        state.config.generations = parseIntKey(text, "generations");
        state.config.dimension = parseIntKey(text, "dimension");
        state.config.crossoverRate = parseDoubleKey(text, "crossover_rate");
        state.config.mutationRate = parseDoubleKey(text, "mutation_rate");
        state.config.bounds.lower = parseDoubleKey(text, "lower");
        state.config.bounds.upper = parseDoubleKey(text, "upper");
        state.config.eliteRatio = parseDoubleKey(text, "elite_ratio");
        state.config.seed = static_cast<unsigned>(parseIntKey(text, "seed"));

        state.generation = static_cast<int>(parseIntKey(text, "generation"));
        state.rngState = parseStringKey(text, "rng_state");

        state.result.bestGenes = parseArrayKey(text, "best_genes");
        state.result.bestFitness = parseDoubleKey(text, "best_fitness");
        state.result.bestHistory = parseArrayKey(text, "best_history");
        state.result.avgHistory = parseArrayKey(text, "avg_history");
        return state;
    }

private:
    template <typename T>
    static void writePod(std::ofstream& out, const T& value) {
        out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    }

    template <typename T>
    static void readPod(std::ifstream& in, T& value) {
        in.read(reinterpret_cast<char*>(&value), sizeof(T));
        if (!in) {
            throw std::runtime_error("Invalid checkpoint format while reading POD value");
        }
    }

    static void writeString(std::ofstream& out, const std::string& s) {
        std::uint64_t size = static_cast<std::uint64_t>(s.size());
        writePod(out, size);
        out.write(s.data(), static_cast<std::streamsize>(size));
    }

    static std::string readString(std::ifstream& in) {
        std::uint64_t size = 0;
        readPod(in, size);
        std::string s(size, '\0');
        in.read(&s[0], static_cast<std::streamsize>(size));
        if (!in) {
            throw std::runtime_error("Invalid checkpoint format while reading string");
        }
        return s;
    }

    static void writeVector(std::ofstream& out, const std::vector<double>& v) {
        std::uint64_t size = static_cast<std::uint64_t>(v.size());
        writePod(out, size);
        if (!v.empty()) {
            out.write(reinterpret_cast<const char*>(v.data()),
                      static_cast<std::streamsize>(sizeof(double) * v.size()));
        }
    }

    static std::vector<double> readVector(std::ifstream& in) {
        std::uint64_t size = 0;
        readPod(in, size);
        std::vector<double> v(static_cast<std::size_t>(size));
        if (!v.empty()) {
            in.read(reinterpret_cast<char*>(v.data()),
                    static_cast<std::streamsize>(sizeof(double) * v.size()));
            if (!in) {
                throw std::runtime_error("Invalid checkpoint format while reading vector");
            }
        }
        return v;
    }

    static std::string vectorToJson(const std::vector<double>& values) {
        std::ostringstream oss;
        oss << "[";
        for (std::size_t i = 0; i < values.size(); ++i) {
            if (i != 0) {
                oss << ", ";
            }
            oss << std::setprecision(std::numeric_limits<double>::max_digits10) << values[i];
        }
        oss << "]";
        return oss.str();
    }

    static std::string escapeJsonString(const std::string& s) {
        std::string out;
        out.reserve(s.size());
        for (char c : s) {
            if (c == '\\' || c == '"') {
                out.push_back('\\');
                out.push_back(c);
            } else if (c == '\n') {
                out += "\\n";
            } else if (c == '\r') {
                out += "\\r";
            } else if (c == '\t') {
                out += "\\t";
            } else {
                out.push_back(c);
            }
        }
        return out;
    }

    static std::string unescapeJsonString(const std::string& s) {
        std::string out;
        out.reserve(s.size());
        for (std::size_t i = 0; i < s.size(); ++i) {
            if (s[i] != '\\') {
                out.push_back(s[i]);
                continue;
            }
            if (i + 1 >= s.size()) {
                throw std::runtime_error("Invalid JSON string escape in checkpoint file");
            }
            const char esc = s[++i];
            switch (esc) {
            case '\\': out.push_back('\\'); break;
            case '"': out.push_back('"'); break;
            case 'n': out.push_back('\n'); break;
            case 'r': out.push_back('\r'); break;
            case 't': out.push_back('\t'); break;
            default:
                throw std::runtime_error("Unsupported JSON escape sequence in checkpoint file");
            }
        }
        return out;
    }

    static std::size_t locateValuePos(const std::string& text, const std::string& key) {
        const std::string token = "\"" + key + "\"";
        const std::size_t k = text.find(token);
        if (k == std::string::npos) {
            throw std::runtime_error("Checkpoint JSON missing key: " + key);
        }
        const std::size_t colon = text.find(':', k + token.size());
        if (colon == std::string::npos) {
            throw std::runtime_error("Checkpoint JSON malformed near key: " + key);
        }
        return colon + 1;
    }

    static void skipWhitespace(const std::string& text, std::size_t& pos) {
        while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos])) != 0) {
            ++pos;
        }
    }

    static double parseNumberAt(const std::string& text, std::size_t pos) {
        skipWhitespace(text, pos);
        const char* begin = text.c_str() + static_cast<std::ptrdiff_t>(pos);
        char* end = nullptr;
        const double value = std::strtod(begin, &end);
        if (end == begin) {
            throw std::runtime_error("Checkpoint JSON expected number value");
        }
        return value;
    }

    static std::uint64_t parseIntKey(const std::string& text, const std::string& key) {
        const std::size_t pos = locateValuePos(text, key);
        const double value = parseNumberAt(text, pos);
        if (value < 0.0) {
            throw std::runtime_error("Checkpoint JSON negative value for unsigned key: " + key);
        }
        return static_cast<std::uint64_t>(value);
    }

    static double parseDoubleKey(const std::string& text, const std::string& key) {
        const std::size_t pos = locateValuePos(text, key);
        return parseNumberAt(text, pos);
    }

    static std::string parseStringKey(const std::string& text, const std::string& key) {
        std::size_t pos = locateValuePos(text, key);
        skipWhitespace(text, pos);
        if (pos >= text.size() || text[pos] != '"') {
            throw std::runtime_error("Checkpoint JSON expected string for key: " + key);
        }
        ++pos;

        std::string raw;
        while (pos < text.size()) {
            const char c = text[pos++];
            if (c == '"') {
                return unescapeJsonString(raw);
            }
            if (c == '\\') {
                if (pos >= text.size()) {
                    throw std::runtime_error("Checkpoint JSON invalid escape for key: " + key);
                }
                raw.push_back(c);
                raw.push_back(text[pos++]);
                continue;
            }
            raw.push_back(c);
        }
        throw std::runtime_error("Checkpoint JSON unterminated string for key: " + key);
    }

    static std::vector<double> parseArrayKey(const std::string& text, const std::string& key) {
        std::size_t pos = locateValuePos(text, key);
        skipWhitespace(text, pos);
        if (pos >= text.size() || text[pos] != '[') {
            throw std::runtime_error("Checkpoint JSON expected array for key: " + key);
        }
        ++pos;

        std::vector<double> out;
        while (pos < text.size()) {
            skipWhitespace(text, pos);
            if (pos < text.size() && text[pos] == ']') {
                ++pos;
                return out;
            }

            const char* begin = text.c_str() + static_cast<std::ptrdiff_t>(pos);
            char* end = nullptr;
            const double value = std::strtod(begin, &end);
            if (end == begin) {
                throw std::runtime_error("Checkpoint JSON invalid array element for key: " + key);
            }
            out.push_back(value);
            pos = static_cast<std::size_t>(end - text.c_str());

            skipWhitespace(text, pos);
            if (pos < text.size() && text[pos] == ',') {
                ++pos;
                continue;
            }
            if (pos < text.size() && text[pos] == ']') {
                ++pos;
                return out;
            }
            throw std::runtime_error("Checkpoint JSON malformed array for key: " + key);
        }
        throw std::runtime_error("Checkpoint JSON unterminated array for key: " + key);
    }
};

} // namespace checkpoint
} // namespace ga
