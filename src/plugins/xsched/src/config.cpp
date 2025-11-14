// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "config.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/core/any.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/util/common_util.hpp"
#include "xsched/properties.hpp"

using namespace ov::xsched_plugin;

namespace {
using DeviceConfig = Configuration::DeviceConfig;
using DeviceGroupConfig = Configuration::DeviceGroupConfig;

const std::vector<std::string> kGroupSkipKeys = {"devices",
                                                 "targets",
                                                 "parameters",
                                                 "params",
                                                 "config",
                                                 "options",
                                                 "strategy",
                                                 "name",
                                                 "label",
                                                 "plugin"};
const std::vector<std::string> kDeviceSkipKeys = {"id",
                                                  "device",
                                                  "name",
                                                  "parameters",
                                                  "params",
                                                  "config",
                                                  "options",
                                                  "properties"};

struct JsonValue {
    enum class Type { Null, Bool, Number, String, Array, Object };

    Type type = Type::Null;
    bool bool_value = false;
    double number_value = 0.0;
    std::string string_value;
    std::vector<JsonValue> array_value;
    std::map<std::string, JsonValue> object_value;

    bool is_null() const { return type == Type::Null; }
    bool is_bool() const { return type == Type::Bool; }
    bool is_number() const { return type == Type::Number; }
    bool is_string() const { return type == Type::String; }
    bool is_array() const { return type == Type::Array; }
    bool is_object() const { return type == Type::Object; }

    const std::vector<JsonValue>& as_array() const { return array_value; }
    const std::map<std::string, JsonValue>& as_object() const { return object_value; }
};

class JsonParser {
public:
    explicit JsonParser(const std::string& source)
        : m_begin(source.c_str()),
          m_current(source.c_str()),
          m_end(m_begin + source.size()) {}

    JsonValue parse() {
        skip_whitespace();
        JsonValue value = parse_value();
        skip_whitespace();
        if (!end()) {
            error("Unexpected characters after JSON value");
        }
        return value;
    }

private:
    const char* m_begin = nullptr;
    const char* m_current = nullptr;
    const char* m_end = nullptr;

    bool end() const { return m_current >= m_end; }

    char peek() const { return end() ? '\0' : *m_current; }

    char get() {
        if (end()) {
            error("Unexpected end of input");
        }
        return *m_current++;
    }

    void skip_whitespace() {
        while (!end() && std::isspace(static_cast<unsigned char>(*m_current))) {
            ++m_current;
        }
    }

    void expect(char expected) {
        char value = get();
        if (value != expected) {
            error(std::string("Expected '") + expected + "'");
        }
    }

    bool match_literal(const char* literal) {
        const char* ptr = m_current;
        while (*literal) {
            if (ptr == m_end || *ptr != *literal) {
                return false;
            }
            ++ptr;
            ++literal;
        }
        if (ptr != m_end) {
            char next = *ptr;
            if (std::isalnum(static_cast<unsigned char>(next)) || next == '_') {
                return false;
            }
        }
        m_current = ptr;
        return true;
    }

    [[noreturn]] void error(const std::string& message) const {
        throw std::runtime_error(message);
    }

    JsonValue parse_value() {
        skip_whitespace();
        if (end()) {
            error("Unexpected end of input");
        }

        char ch = peek();
        if (ch == '{') {
            return parse_object();
        }
        if (ch == '[') {
            return parse_array();
        }
        if (ch == '"') {
            return parse_string_value();
        }
        if (ch == '-' || std::isdigit(static_cast<unsigned char>(ch))) {
            return parse_number_value();
        }
        if (ch == 't' && match_literal("true")) {
            JsonValue value;
            value.type = JsonValue::Type::Bool;
            value.bool_value = true;
            return value;
        }
        if (ch == 'f' && match_literal("false")) {
            JsonValue value;
            value.type = JsonValue::Type::Bool;
            value.bool_value = false;
            return value;
        }
        if (ch == 'n' && match_literal("null")) {
            return JsonValue{};
        }

        error("Unexpected character in JSON value");
        return JsonValue{};
    }

    JsonValue parse_object() {
        JsonValue object;
        object.type = JsonValue::Type::Object;
        expect('{');
        skip_whitespace();
        if (peek() == '}') {
            get();
            return object;
        }
        while (true) {
            skip_whitespace();
            if (peek() != '"') {
                error("Expected string key in JSON object");
            }
            std::string key = parse_string();
            skip_whitespace();
            expect(':');
            JsonValue value = parse_value();
            object.object_value.emplace(std::move(key), std::move(value));
            skip_whitespace();
            char delimiter = get();
            if (delimiter == '}') {
                break;
            }
            if (delimiter != ',') {
                error("Expected ',' or '}' in JSON object");
            }
            skip_whitespace();
        }
        return object;
    }

    JsonValue parse_array() {
        JsonValue array;
        array.type = JsonValue::Type::Array;
        expect('[');
        skip_whitespace();
        if (peek() == ']') {
            get();
            return array;
        }
        while (true) {
            JsonValue value = parse_value();
            array.array_value.push_back(std::move(value));
            skip_whitespace();
            char delimiter = get();
            if (delimiter == ']') {
                break;
            }
            if (delimiter != ',') {
                error("Expected ',' or ']' in JSON array");
            }
            skip_whitespace();
        }
        return array;
    }

    JsonValue parse_string_value() {
        JsonValue value;
        value.type = JsonValue::Type::String;
        value.string_value = parse_string();
        return value;
    }

    JsonValue parse_number_value() {
        const char* start = m_current;
        if (peek() == '-') {
            ++m_current;
        }
        if (end()) {
            error("Invalid number literal");
        }
        if (peek() == '0') {
            ++m_current;
        } else {
            if (!std::isdigit(static_cast<unsigned char>(peek()))) {
                error("Invalid number literal");
            }
            while (!end() && std::isdigit(static_cast<unsigned char>(peek()))) {
                ++m_current;
            }
        }
        if (!end() && peek() == '.') {
            ++m_current;
            if (end() || !std::isdigit(static_cast<unsigned char>(peek()))) {
                error("Invalid number literal");
            }
            while (!end() && std::isdigit(static_cast<unsigned char>(peek()))) {
                ++m_current;
            }
        }
        if (!end() && (peek() == 'e' || peek() == 'E')) {
            ++m_current;
            if (!end() && (peek() == '+' || peek() == '-')) {
                ++m_current;
            }
            if (end() || !std::isdigit(static_cast<unsigned char>(peek()))) {
                error("Invalid number literal");
            }
            while (!end() && std::isdigit(static_cast<unsigned char>(peek()))) {
                ++m_current;
            }
        }

        const char* finish = m_current;
        std::string buffer(start, finish);
        char* end_ptr = nullptr;
        double number = std::strtod(buffer.c_str(), &end_ptr);
        if (end_ptr == buffer.c_str()) {
            error("Invalid number literal");
        }

        JsonValue value;
        value.type = JsonValue::Type::Number;
        value.number_value = number;
        return value;
    }

    std::string parse_string() {
        std::string result;
        expect('"');
        while (true) {
            if (end()) {
                error("Unterminated string literal");
            }
            char ch = get();
            if (ch == '"') {
                break;
            }
            if (ch == '\\') {
                if (end()) {
                    error("Incomplete escape sequence");
                }
                char esc = get();
                switch (esc) {
                case '"':
                    result.push_back('"');
                    break;
                case '\\':
                    result.push_back('\\');
                    break;
                case '/':
                    result.push_back('/');
                    break;
                case 'b':
                    result.push_back('\b');
                    break;
                case 'f':
                    result.push_back('\f');
                    break;
                case 'n':
                    result.push_back('\n');
                    break;
                case 'r':
                    result.push_back('\r');
                    break;
                case 't':
                    result.push_back('\t');
                    break;
                case 'u': {
                    uint32_t codepoint = parse_hex4();
                    if (codepoint >= 0xD800 && codepoint <= 0xDBFF) {
                        if (!(peek() == '\\')) {
                            error("Invalid unicode surrogate pair");
                        }
                        get();
                        if (get() != 'u') {
                            error("Invalid unicode surrogate pair");
                        }
                        uint32_t low = parse_hex4();
                        if (low < 0xDC00 || low > 0xDFFF) {
                            error("Invalid unicode surrogate pair");
                        }
                        codepoint = 0x10000 + ((codepoint - 0xD800) << 10) + (low - 0xDC00);
                    }
                    append_codepoint(result, codepoint);
                    break;
                }
                default:
                    error("Invalid escape sequence");
                }
            } else {
                if (static_cast<unsigned char>(ch) < 0x20) {
                    error("Invalid control character in string literal");
                }
                result.push_back(ch);
            }
        }
        return result;
    }

    uint32_t parse_hex4() {
        uint32_t value = 0;
        for (int i = 0; i < 4; ++i) {
            char ch = get();
            value <<= 4;
            if (ch >= '0' && ch <= '9') {
                value |= static_cast<uint32_t>(ch - '0');
            } else if (ch >= 'a' && ch <= 'f') {
                value |= static_cast<uint32_t>(10 + ch - 'a');
            } else if (ch >= 'A' && ch <= 'F') {
                value |= static_cast<uint32_t>(10 + ch - 'A');
            } else {
                error("Invalid unicode escape sequence");
            }
        }
        return value;
    }

    static void append_codepoint(std::string& out, uint32_t codepoint) {
        if (codepoint <= 0x7F) {
            out.push_back(static_cast<char>(codepoint));
        } else if (codepoint <= 0x7FF) {
            out.push_back(static_cast<char>(0xC0 | ((codepoint >> 6) & 0x1F)));
            out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
        } else if (codepoint <= 0xFFFF) {
            out.push_back(static_cast<char>(0xE0 | ((codepoint >> 12) & 0x0F)));
            out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xF0 | ((codepoint >> 18) & 0x07)));
            out.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
        }
    }
};

std::string escape_json_string(const std::string& input) {
    std::string result;
    result.reserve(input.size() + 2);
    result.push_back('"');
    for (char ch : input) {
        switch (ch) {
        case '"':
            result += "\\\"";
            break;
        case '\\':
            result += "\\\\";
            break;
        case '\b':
            result += "\\b";
            break;
        case '\f':
            result += "\\f";
            break;
        case '\n':
            result += "\\n";
            break;
        case '\r':
            result += "\\r";
            break;
        case '\t':
            result += "\\t";
            break;
        default:
            if (static_cast<unsigned char>(ch) < 0x20) {
                static const char* digits = "0123456789ABCDEF";
                char buffer[7] = {'\\', 'u', '0', '0', '0', '0', '\0'};
                unsigned char value = static_cast<unsigned char>(ch);
                buffer[4] = digits[(value >> 4) & 0x0F];
                buffer[5] = digits[value & 0x0F];
                result += buffer;
            } else {
                result.push_back(ch);
            }
        }
    }
    result.push_back('"');
    return result;
}

std::string json_to_compact_string(const JsonValue& value) {
    switch (value.type) {
    case JsonValue::Type::Null:
        return "null";
    case JsonValue::Type::Bool:
        return value.bool_value ? "true" : "false";
    case JsonValue::Type::Number: {
        std::ostringstream oss;
        oss << std::setprecision(std::numeric_limits<double>::digits10 + 1) << value.number_value;
        return ov::util::trim(oss.str());
    }
    case JsonValue::Type::String:
        return escape_json_string(value.string_value);
    case JsonValue::Type::Array: {
        std::string result = "[";
        bool first = true;
        for (const auto& item : value.as_array()) {
            if (!first) {
                result.push_back(',');
            }
            first = false;
            result += json_to_compact_string(item);
        }
        result.push_back(']');
        return result;
    }
    case JsonValue::Type::Object: {
        std::string result = "{";
        bool first = true;
        for (const auto& kv : value.as_object()) {
            if (!first) {
                result.push_back(',');
            }
            first = false;
            result += escape_json_string(kv.first);
            result.push_back(':');
            result += json_to_compact_string(kv.second);
        }
        result.push_back('}');
        return result;
    }
    }
    return {};
}

std::string json_value_to_string(const JsonValue& value) {
    if (value.is_string()) {
        return value.string_value;
    }
    if (value.is_bool()) {
        return value.bool_value ? "true" : "false";
    }
    if (value.is_number()) {
        double number = value.number_value;
        if (std::isfinite(number)) {
            double rounded = std::round(number);
            if (std::fabs(number - rounded) < 1e-9 &&
                std::fabs(rounded) <= static_cast<double>(std::numeric_limits<long long>::max())) {
                return std::to_string(static_cast<long long>(rounded));
            }
        }
        std::ostringstream oss;
        oss << std::setprecision(std::numeric_limits<double>::digits10 + 1) << number;
        return ov::util::trim(oss.str());
    }
    if (value.is_null()) {
        return std::string{};
    }
    return json_to_compact_string(value);
}

const JsonValue* find_member(const JsonValue& node, const std::string& key) {
    if (!node.is_object()) {
        return nullptr;
    }
    const auto& object = node.as_object();
    auto it = object.find(key);
    if (it == object.end()) {
        return nullptr;
    }
    return &it->second;
}

std::map<std::string, std::string> json_to_map(const JsonValue& node, const std::vector<std::string>& skip) {
    std::map<std::string, std::string> result;
    if (!node.is_object()) {
        return result;
    }
    for (const auto& kv : node.as_object()) {
        if (std::find(skip.begin(), skip.end(), kv.first) != skip.end()) {
            continue;
        }
        if (kv.second.is_array() || kv.second.is_object()) {
            continue;
        }
        result[kv.first] = json_value_to_string(kv.second);
    }
    return result;
}

void merge_secondary_map(std::map<std::string, std::string>& destination, const JsonValue& maybe_map) {
    if (!maybe_map.is_object()) {
        return;
    }
    auto values = json_to_map(maybe_map, {});
    destination.insert(values.begin(), values.end());
}

DeviceConfig parse_device_entry(const JsonValue& node, bool throw_on_error, const std::string& fallback_id = {}) {
    DeviceConfig device;
    if (node.is_string()) {
        device.id = ov::util::trim(node.string_value);
        return device;
    }
    if (!node.is_object()) {
        if (throw_on_error) {
            OPENVINO_THROW("Unsupported device entry format in ov::device::priorities value: ", json_to_compact_string(node));
        }
        return device;
    }
    if (const auto* id = find_member(node, "id"); id && id->is_string()) {
        device.id = ov::util::trim(id->string_value);
    } else if (const auto* dev = find_member(node, "device"); dev && dev->is_string()) {
        device.id = ov::util::trim(dev->string_value);
    } else if (const auto* name = find_member(node, "name"); name && name->is_string()) {
        device.id = ov::util::trim(name->string_value);
    }

    device.parameters = json_to_map(node, kDeviceSkipKeys);
    if (const auto* params = find_member(node, "parameters")) {
        merge_secondary_map(device.parameters, *params);
    }
    if (const auto* params = find_member(node, "params")) {
        merge_secondary_map(device.parameters, *params);
    }
    if (const auto* cfg = find_member(node, "config")) {
        merge_secondary_map(device.parameters, *cfg);
    }
    if (const auto* opts = find_member(node, "options")) {
        merge_secondary_map(device.parameters, *opts);
    }
    if (const auto* props = find_member(node, "properties")) {
        merge_secondary_map(device.parameters, *props);
    }

    if (device.id.empty() && !fallback_id.empty()) {
        device.id = ov::util::trim(fallback_id);
    }

    if (device.id.empty() && throw_on_error) {
        OPENVINO_THROW("Device entry in ov::device::priorities is missing identifier: ", json_to_compact_string(node));
    }

    return device;
}

void append_devices(const JsonValue& node, DeviceGroupConfig& group, bool throw_on_error) {
    if (node.is_null()) {
        return;
    }
    if (node.is_string()) {
        const auto entries = ov::util::split(node.string_value, ',', true);
        for (const auto& entry : entries) {
            if (entry.empty()) {
                continue;
            }
            DeviceConfig device;
            device.id = entry;
            group.devices.push_back(std::move(device));
        }
        return;
    }
    if (node.is_array()) {
        for (const auto& item : node.as_array()) {
            auto device = parse_device_entry(item, throw_on_error);
            if (!device.id.empty()) {
                group.devices.push_back(std::move(device));
            }
        }
        return;
    }
    if (node.is_object()) {
        for (const auto& kv : node.as_object()) {
            if (std::find(kGroupSkipKeys.begin(), kGroupSkipKeys.end(), kv.first) != kGroupSkipKeys.end()) {
                continue;
            }
            if (!kv.second.is_object()) {
                DeviceConfig device;
                device.id = ov::util::trim(kv.first);
                if (!device.id.empty()) {
                    if (!kv.second.is_null()) {
                        device.parameters["value"] = json_value_to_string(kv.second);
                    }
                    group.devices.push_back(std::move(device));
                }
                continue;
            }

            auto device = parse_device_entry(kv.second, throw_on_error, kv.first);
            if (!device.id.empty()) {
                group.devices.push_back(std::move(device));
            }
        }
        return;
    }

    if (throw_on_error) {
        OPENVINO_THROW("Unsupported devices declaration format in ov::device::priorities value: ", json_to_compact_string(node));
    }
}

DeviceGroupConfig parse_group(const JsonValue& node, const std::string& label_hint, bool throw_on_error) {
    DeviceGroupConfig group;
    const auto label_hint_trimmed = ov::util::trim(label_hint);

    if (node.is_string()) {
        group.label = label_hint_trimmed;
        const auto entries = ov::util::split(node.string_value, ',', true);
        for (const auto& entry : entries) {
            if (entry.empty()) {
                continue;
            }
            DeviceConfig device;
            device.id = entry;
            group.devices.push_back(std::move(device));
        }
        return group;
    }

    if (node.is_array()) {
        if (group.label.empty() && !label_hint_trimmed.empty()) {
            group.label = label_hint_trimmed;
        }
        for (const auto& item : node.as_array()) {
            auto device = parse_device_entry(item, throw_on_error);
            if (!device.id.empty()) {
                group.devices.push_back(std::move(device));
            }
        }
        return group;
    }

    if (!node.is_object()) {
        if (throw_on_error) {
            OPENVINO_THROW("Unsupported group format for ov::device::priorities value: ", json_to_compact_string(node));
        }
        return group;
    }

    if (const auto* name = find_member(node, "name"); name && name->is_string()) {
        group.label = ov::util::trim(name->string_value);
    }
    if (group.label.empty()) {
        if (const auto* label_node = find_member(node, "label"); label_node && label_node->is_string()) {
            group.label = ov::util::trim(label_node->string_value);
        }
    }
    if (group.label.empty() && !label_hint.empty()) {
        group.label = label_hint;
    }
    if (group.label.empty()) {
        if (const auto* plugin = find_member(node, "plugin"); plugin && plugin->is_string()) {
            group.label = ov::util::trim(plugin->string_value);
        }
    }

    if (const auto* strategy = find_member(node, "strategy"); strategy && strategy->is_string()) {
        group.strategy = ov::util::trim(strategy->string_value);
    }

    group.parameters = json_to_map(node, kGroupSkipKeys);
    if (const auto* params = find_member(node, "parameters")) {
        merge_secondary_map(group.parameters, *params);
    }
    if (const auto* params = find_member(node, "params")) {
        merge_secondary_map(group.parameters, *params);
    }
    if (const auto* cfg = find_member(node, "config")) {
        merge_secondary_map(group.parameters, *cfg);
    }
    if (const auto* opts = find_member(node, "options")) {
        merge_secondary_map(group.parameters, *opts);
    }

    const JsonValue* devices_node = nullptr;
    if (const auto* devices = find_member(node, "devices")) {
        devices_node = devices;
    } else if (const auto* targets = find_member(node, "targets")) {
        devices_node = targets;
    }

    if (!devices_node) {
        append_devices(node, group, throw_on_error);
    } else {
        append_devices(*devices_node, group, throw_on_error);
    }

    if (group.devices.empty() && throw_on_error) {
        OPENVINO_THROW("No devices defined for group in ov::device::priorities value: ", json_to_compact_string(node));
    }

    return group;
}

std::vector<DeviceGroupConfig> parse_json_priorities(const JsonValue& root, bool throw_on_error) {
    std::vector<DeviceGroupConfig> groups;

    if (root.is_array()) {
        for (const auto& item : root.as_array()) {
            auto group = parse_group(item, std::string{}, throw_on_error);
            if (!group.devices.empty()) {
                groups.push_back(std::move(group));
            }
        }
        return groups;
    }

    if (!root.is_object()) {
        if (throw_on_error) {
            OPENVINO_THROW("ov::device::priorities JSON value must be an object or array");
        }
        return groups;
    }

    if (const auto* groups_node = find_member(root, "groups")) {
        if (!groups_node->is_array()) {
            if (throw_on_error) {
                OPENVINO_THROW("ov::device::priorities JSON 'groups' field must be an array");
            }
        } else {
            for (const auto& item : groups_node->as_array()) {
                auto group = parse_group(item, std::string{}, throw_on_error);
                if (!group.devices.empty()) {
                    groups.push_back(std::move(group));
                }
            }
        }
        return groups;
    }

    if (root.as_object().count("devices") || root.as_object().count("targets") || root.as_object().count("strategy")) {
        auto group = parse_group(root, std::string{}, throw_on_error);
        if (!group.devices.empty()) {
            groups.push_back(std::move(group));
        }
        return groups;
    }

    for (const auto& kv : root.as_object()) {
        auto group = parse_group(kv.second, kv.first, throw_on_error);
        if (!group.devices.empty()) {
            groups.push_back(std::move(group));
        }
    }

    return groups;
}

std::vector<DeviceGroupConfig> parse_simple_priorities(const std::string& value) {
    std::vector<DeviceGroupConfig> groups;
    if (value.empty()) {
        return groups;
    }

    auto group_tokens = ov::util::split(value, ';', true);
    if (group_tokens.empty()) {
        group_tokens.push_back(value);
    }

    for (const auto& token : group_tokens) {
        if (token.empty()) {
            continue;
        }

        DeviceGroupConfig group;
        std::string remainder = token;
        auto colon_pos = remainder.find(':');
        if (colon_pos != std::string::npos) {
            group.label = ov::util::trim(remainder.substr(0, colon_pos));
            remainder = remainder.substr(colon_pos + 1);
        }

        auto segments = ov::util::split(remainder, '|', true);
        if (segments.empty()) {
            continue;
        }

        auto device_segment = segments.front();
        auto devices = ov::util::split(device_segment, ',', true);
        for (const auto& dev : devices) {
            if (dev.empty()) {
                continue;
            }
            DeviceConfig device;
            device.id = dev;
            group.devices.push_back(std::move(device));
        }

        for (size_t i = 1; i < segments.size(); ++i) {
            const auto& segment = segments[i];
            if (segment.empty()) {
                continue;
            }
            auto eq_pos = segment.find('=');
            if (eq_pos == std::string::npos) {
                if (group.strategy.empty()) {
                    group.strategy = ov::util::trim(segment);
                } else {
                    group.parameters["extra" + std::to_string(i)] = ov::util::trim(segment);
                }
                continue;
            }
            auto key = ov::util::trim(segment.substr(0, eq_pos));
            auto val = ov::util::trim(segment.substr(eq_pos + 1));
            if (ov::util::to_lower(key) == "strategy" && group.strategy.empty()) {
                group.strategy = val;
            } else {
                group.parameters[key] = val;
            }
        }

        if (!group.devices.empty()) {
            groups.push_back(std::move(group));
        }
    }

    return groups;
}
}  // namespace

Configuration::Configuration() {}

Configuration::Configuration(const ov::AnyMap& config, const Configuration& defaultCfg, bool throwOnUnsupported) {
    *this = defaultCfg;
    for (auto&& [key, value] : config) {
        if (ov::xsched_plugin::disable_transformations == key) {
            disable_transformations = value.as<bool>();
        } else if (ov::internal::exclusive_async_requests == key) {
            exclusive_async_requests = value.as<bool>();
        } else if (ov::num_streams.name() == key) {
            ov::Any val = value.as<std::string>();
            auto streams_value = val.as<ov::streams::Num>();
            if (streams_value.num >= 0) {
                streams = streams_value.num;
            } else if (streams_value == ov::streams::NUMA) {
                streams = 1;
            } else if (streams_value == ov::streams::AUTO) {
                streams = ov::threading::IStreamsExecutor::Config::get_default_num_streams();
            } else {
                OPENVINO_THROW("Wrong value for property key ",
                               key,
                               ". Expected non negative numbers (#streams) or ",
                               "ov::streams::NUMA|ov::streams::AUTO, Got: ",
                               value.as<std::string>());
            }
        } else if (ov::inference_num_threads.name() == key) {
            int val;
            try {
                val = value.as<int>();
            } catch (const std::exception&) {
                OPENVINO_THROW("Wrong value for property key ", key, ". Expected only positive numbers (#threads)");
            }
            if (val < 0) {
                OPENVINO_THROW("Wrong value for property key ", key, ". Expected only positive numbers (#threads)");
            }
            threads = val;
        } else if (ov::internal::threads_per_stream.name() == key) {
            try {
                threads_per_stream = value.as<int>();
            } catch (const std::exception&) {
                OPENVINO_THROW("Wrong value ", value.as<std::string>(), "for property key ", key);
            }
        } else if (ov::device::id == key) {
            device_id = std::stoi(value.as<std::string>());
            OPENVINO_ASSERT(device_id <= 0, "Device ID ", device_id, " is not supported");
        } else if (ov::device::priorities == key) {
            set_device_priorities(value, throwOnUnsupported);
        } else if (ov::enable_profiling == key) {
            perf_count = value.as<bool>();
        } else if (ov::hint::performance_mode == key) {
            std::stringstream strm{value.as<std::string>()};
            strm >> performance_mode;
        } else if (ov::hint::inference_precision == key) {
            inference_precision = value.as<ov::element::Type>();
        } else if (ov::hint::execution_mode == key) {
            execution_mode = value.as<ov::hint::ExecutionMode>();
            if ((execution_mode != ov::hint::ExecutionMode::ACCURACY) &&
                (execution_mode != ov::hint::ExecutionMode::PERFORMANCE)) {
                OPENVINO_THROW("Unsupported execution mode, should be ACCURACY or PERFORMANCE, but was: ",
                               value.as<std::string>());
            }
        } else if (ov::hint::num_requests == key) {
            const auto& tmp_val = value.as<std::string>();
            int tmp_i = std::stoi(tmp_val);
            if (tmp_i >= 0)
                num_requests = tmp_i;
            else
                OPENVINO_THROW("Incorrect value, it should be unsigned integer: ", key);
        } else if (ov::log::level == key) {
            log_level = value.as<ov::log::Level>();
        } else if (ov::hint::model_priority == key) {
            model_priority = value.as<ov::hint::Priority>();
        } else if (ov::cache_encryption_callbacks == key) {
            encryption_callbacks = value.as<EncryptionCallbacks>();
        } else if (ov::hint::scheduling_core_type == key) {
            try {
                schedulingCoreType = value.as<ov::hint::SchedulingCoreType>();
            } catch (const std::exception&) {
                OPENVINO_THROW("Wrong value ", value.as<ov::hint::SchedulingCoreType>(), "for property key ", key);
            }
        } else if (key == ov::hint::enable_hyper_threading) {
            try {
                enableHyperThreading = value.as<bool>();
            } catch (ov::Exception&) {
                OPENVINO_THROW("Wrong value ",
                               value.as<std::string>(),
                               "for property key ",
                               key,
                               ". Expected only true/false.");
            }
        } else if (key == ov::hint::enable_cpu_pinning.name()) {
            try {
                enableCpuPinning = value.as<bool>();
            } catch (ov::Exception&) {
                OPENVINO_THROW("Wrong value ",
                               value.as<std::string>(),
                               "for property key ",
                               key,
                               ". Expected only true/false.");
            }
        } else if (ov::compilation_num_threads.name() == key) {
            int val;
            try {
                val = value.as<int>();
            } catch (const std::exception&) {
                OPENVINO_THROW("Wrong value for property key ", key, ". Expected only positive numbers (#threads)");
            }
            if (val < 0) {
                OPENVINO_THROW("Wrong value for property key ", key, ". Expected only positive numbers (#threads)");
            }
            compilation_thread_num = val;
        } else if (ov::weights_path == key) {
            weights_path = value.as<std::string>();
            if (!weights_path.empty()) {
                compiled_model_runtime_properties[ov::weights_path.name()] = weights_path.string();
            }
        } else if (ov::cache_mode == key) {
            cache_mode = value.as<CacheMode>();
        } else if (throwOnUnsupported) {
            OPENVINO_THROW("Property was not found: ", key);
        }
    }
}

ov::Any Configuration::Get(const std::string& name) const {
    if (name == ov::device::id) {
        return {std::to_string(device_id)};
    } else if (name == ov::enable_profiling) {
        return {perf_count};
    } else if (name == ov::internal::exclusive_async_requests) {
        return {exclusive_async_requests};
    } else if (name == ov::xsched_plugin::disable_transformations) {
        return {disable_transformations};
    } else if (name == ov::num_streams) {
        return {std::to_string(streams)};
    } else if (name == ov::inference_num_threads) {
        return {std::to_string(threads)};
    } else if (name == ov::internal::threads_per_stream) {
        return {std::to_string(threads_per_stream)};
    } else if (name == ov::hint::performance_mode) {
        return performance_mode;
    } else if (name == ov::hint::inference_precision) {
        return inference_precision;
    } else if (name == ov::hint::execution_mode) {
        return execution_mode;
    } else if (name == ov::hint::num_requests) {
        return num_requests;
    } else if (name == ov::log::level) {
        return log_level;
    } else if (name == ov::hint::model_priority) {
        return model_priority;
    } else if (name == ov::compilation_num_threads) {
        return {std::to_string(compilation_thread_num)};
    } else if (name == ov::hint::scheduling_core_type) {
        return schedulingCoreType;
    } else if (name == ov::hint::enable_cpu_pinning) {
        return enableCpuPinning;
    } else if (name == ov::hint::enable_hyper_threading) {
        return enableHyperThreading;
    } else if (name == ov::weights_path) {
        return weights_path.string();
    } else if (name == ov::internal::compiled_model_runtime_properties) {
        return compiled_model_runtime_properties;
    } else if (name == ov::cache_mode) {
        return cache_mode;
    } else if (name == ov::device::priorities) {
        return device_priorities_raw;
    } else {
        OPENVINO_THROW("Property was not found: ", name);
    }
}

void Configuration::set_device_priorities(const ov::Any& value, bool throwOnUnsupported) {
    device_priorities_raw.clear();
    device_priority_groups.clear();

    if (value.empty()) {
        return;
    }

    auto handle_string = [&](const std::string& serialized) {
        parse_device_priorities_string(serialized, throwOnUnsupported);
    };

    try {
        if (value.is<std::string>()) {
            handle_string(value.as<std::string>());
            return;
        }
    } catch (const std::exception& ex) {
        if (throwOnUnsupported) {
            OPENVINO_THROW("Wrong value for property key ", ov::device::priorities.name(), ": ", ex.what());
        }
        return;
    }

    try {
        if (value.is<const char*>()) {
            handle_string(std::string{value.as<const char*>()});
            return;
        }
    } catch (const std::exception& ex) {
        if (throwOnUnsupported) {
            OPENVINO_THROW("Wrong value for property key ", ov::device::priorities.name(), ": ", ex.what());
        }
        return;
    }

    try {
        if (value.is<std::vector<std::string>>()) {
            const auto raw = ov::util::join(value.as<std::vector<std::string>>(), ";");
            handle_string(raw);
            return;
        }
    } catch (const std::exception& ex) {
        if (throwOnUnsupported) {
            OPENVINO_THROW("Wrong value for property key ", ov::device::priorities.name(), ": ", ex.what());
        }
        return;
    }

    try {
        if (value.is<std::vector<const char*>>()) {
            const auto raw_input = value.as<std::vector<const char*>>();
            std::vector<std::string> normalized;
            normalized.reserve(raw_input.size());
            for (auto ptr : raw_input) {
                if (ptr) {
                    normalized.emplace_back(ptr);
                }
            }
            const auto raw = ov::util::join(normalized, ";");
            handle_string(raw);
            return;
        }
    } catch (const std::exception& ex) {
        if (throwOnUnsupported) {
            OPENVINO_THROW("Wrong value for property key ", ov::device::priorities.name(), ": ", ex.what());
        }
        return;
    }

    try {
        handle_string(value.as<std::string>());
    } catch (const std::exception& ex) {
        if (throwOnUnsupported) {
            OPENVINO_THROW("Wrong value for property key ", ov::device::priorities.name(), ": ", ex.what());
        }
    }
}

void Configuration::parse_device_priorities_string(const std::string& raw, bool throwOnUnsupported) {
    device_priorities_raw = ov::util::trim(raw);
    device_priority_groups.clear();

    if (device_priorities_raw.empty()) {
        return;
    }

    const auto first_char = device_priorities_raw.front();
    if (first_char == '{' || first_char == '[') {
        try {
            JsonParser parser{device_priorities_raw};
            auto json = parser.parse();
            device_priority_groups = parse_json_priorities(json, throwOnUnsupported);
            if (!device_priority_groups.empty() || !throwOnUnsupported) {
                return;
            }
        } catch (const std::exception& ex) {
            if (throwOnUnsupported) {
                OPENVINO_THROW("Failed to parse ov::device::priorities JSON value: ", ex.what());
            }
        }
    }

    auto simple = parse_simple_priorities(device_priorities_raw);
    if (simple.empty() && throwOnUnsupported) {
        OPENVINO_THROW("Failed to parse ov::device::priorities value: ", device_priorities_raw);
    }
    device_priority_groups = std::move(simple);
}
