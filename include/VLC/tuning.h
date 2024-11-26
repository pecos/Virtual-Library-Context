#ifndef _VLC_TUNING_H_
#define _VLC_TUNING_H_
/**
 * This header is used to automatically finding the best config for VLCs.
 * 
 * Methods here are used to generate information used by driver 
 * which do grid search on different resouce configurations.
 * 
 * please install RapidJSON before use this tool.
 * for ubuntu:
 *     apt install rapidjson-dev
 */
#include <vector>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <rapidjson/document.h>

#include "VLC/loader.h"

// create and copy a new char array
static char *mov_str(const char * in) {
    char* out = new char[strlen(in) + 1]; 
    strcpy(out, in);
    return out;
}

namespace VLC {
    /**
     * Config for a VLCs tuning task.
     * 
     * Not copiable, pass it by std::move or pointer.
     */
    class TuningConfig {
    public:
        char * name;
        char * entry_point_symbol;
        char * path;
        char **argv;
        int argc;

        TuningConfig() = default;

        TuningConfig(const TuningConfig&) = delete;

        TuningConfig(TuningConfig&& obj) {
            name = obj.name;
            entry_point_symbol = obj.entry_point_symbol;
            path = obj.path;
            argv = obj.argv;
            argc = obj.argc;

            obj.name = nullptr;
            obj.entry_point_symbol = nullptr;
            obj.path = nullptr;
            obj.argv = nullptr;
        }

        ~TuningConfig() {
            delete name;
            delete entry_point_symbol;
            delete path;
            if (argv) {
                for (int i = 0; i < argc; i++) {
                    delete argv[i];
                } 
                delete argv;
            }
        };
    };

    /**
     * Given a json file, parse the config in it.
     * Return a list of TuningConfig object.
     */
    std::vector<TuningConfig>
    parse_config(const char *config_file_path, const char *arg_zero) {        
        std::ifstream file(config_file_path);
        if (!file.is_open()) {
            VLC_DIE("VLC: cannot open config file on path: %s", config_file_path);
        }
        std::ostringstream buffer;
        buffer << file.rdbuf();
        const std::string jsonStr = buffer.str();

        // parse json file into DOM
        rapidjson::Document document;
        document.Parse(jsonStr.c_str());

        if (document.HasParseError()) {
            VLC_DIE("VLC: cannot parse JSON file: %s", config_file_path);
        }

        // initialize output dictionary
        std::vector<TuningConfig> output;
        std::unordered_map<std::string, std::string> names;

        for (rapidjson::SizeType i = 0; i < document.Size(); i++) {
            const rapidjson::Value& vlc = document[i];

            TuningConfig config;
            config.name = mov_str(vlc["name"].GetString());
            config.entry_point_symbol = mov_str(vlc["entry_point_symbol"].GetString());
            config.path = mov_str(vlc["path"].GetString());

            // parse argv
            const rapidjson::Value& argv = vlc["argv"];
            config.argc = argv.Size() + 1;
            char **arg_array = new char *[config.argc];
            config.argv = arg_array;

            // the first arg is the executable
            arg_array[0] = mov_str(arg_zero);

            for (rapidjson::SizeType i = 0; i < argv.Size(); i++) {
                arg_array[i + 1] = mov_str(argv[i].GetString());
            }

            // register entry point symbols for each VLCs
            names[std::string(config.name)] = std::string(config.entry_point_symbol);

            output.push_back(std::move(config));
        }
        VLC::Loader::register_func_names(names);

        return output; // move construcotr is happened here
    }
}

#endif // _VLC_TUNING_H_