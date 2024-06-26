#include "common.hpp"
#include "../cxxopts/include/cxxopts.hpp"

int main(int argc, char* argv[]) {

    if (argc < 1) {
        std::cerr << "Error: no command-line arguments were provided." << std::endl;
        return 1;
    }

    std::string silk_filename = argv[0];
    std::string nnue_filename = silk_filename + "_nnue.dat";

    // define argument parser
    cxxopts::Options options("silk", "A CUDA drifter searcher");

    options.add_options()

    // positional arguments (obligatory)
    ("input_filename", "LifeHistory RLE specifying the problem", cxxopts::value<std::string>())
    ("max_active_width", "maximum width of active region", cxxopts::value<int>())
    ("max_active_height", "maximum height of active region", cxxopts::value<int>())
    ("max_active_cells", "maximum number of active cells", cxxopts::value<int>())

    // optional arguments
    ("cadicals", "number of CaDiCaL threads to stabilise results", cxxopts::value<int>()->default_value("8"))
    ("p,period", "minimum period of oscillators to report", cxxopts::value<int>()->default_value("999999999"))
    ("d,dataset", "filename of dataset to output", cxxopts::value<std::string>()->default_value(""))
    ("m,max-perturbed-time", "maximum time the catalyst can interact", cxxopts::value<int>()->default_value("999999999"))            
    ("s,min-stable", "minimum unclean catalyst stable time before report", cxxopts::value<int>()->default_value("999999999"))
    ("e,exempt-existing", "whether to ignore constraints on ZOI of input catalyst cells", cxxopts::value<bool>()->default_value("false"))      

    // help
    ("h,help", "Print usage");

    options.parse_positional({"input_filename", "max_active_width", "max_active_height", "max_active_cells"});
    options.positional_help("path/to/input.rle max_active_width max_active_height max_active_cells");

    // apply argument parser to cmdline args
    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        // print help message and exit
        std::cerr << options.help() << std::endl;
        return 0;
    }

    // extract positional arguments
    std::string input_filename = result["input_filename"].as<std::string>();
    int active_width = result["max_active_width"].as<int>();
    int active_height = result["max_active_height"].as<int>();
    int active_pop = result["max_active_cells"].as<int>();

    // extract optional arguments
    int num_cadical_threads = result["cadicals"].as<int>();
    int min_report_period = result["period"].as<int>();
    std::string dataset_filename = result["dataset"].as<std::string>();
    int max_perturbed_time = result["max-perturbed-time"].as<int>();
    int min_stable = result["min-stable"].as<int>();
    bool exempt_existing = result["exempt-existing"].as<bool>();

    std::cerr << "Info: Silk invoked as " << silk_filename << std::endl;

    // run program
    int return_code = silk_main(
        active_width,
        active_height,
        active_pop,
        input_filename,
        nnue_filename,
        num_cadical_threads,
        min_report_period,
        max_perturbed_time,
        min_stable,
        exempt_existing,
        dataset_filename
    );

    return return_code;
}

