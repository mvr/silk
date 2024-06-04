#include "common.hpp"

int main(int argc, char* argv[]) {

    if (argc < 1) {
        std::cerr << "Error: no command-line arguments were provided." << std::endl;
        return 1;
    }

    std::string silk_filename = argv[0];
    std::string nnue_filename = silk_filename + "_nnue.dat";
    std::cerr << "Info: Silk invoked as " << silk_filename << std::endl;

    // ***** PARSE ARGUMENTS *****

    int active_width = 8;
    int active_height = 8;
    int active_pop = 16;

    std::string input_filename = "examples/eater.rle";
    int num_cadical_threads = 8;

    int return_code = silk_main(
        active_width,
        active_height,
        active_pop,
        input_filename,
        nnue_filename,
        num_cadical_threads
    );

    return return_code;
}

