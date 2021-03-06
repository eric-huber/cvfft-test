#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <boost/program_options.hpp>

#include <cstdlib>
#include <chrono>
#include <iostream>
#include <fstream>
#include <iomanip>
 
using namespace std;
using namespace cv;
using namespace chrono;

namespace po = boost::program_options;

const char*     _data_file_name = "fft-data.txt";
const char*     _fft_file_name  = "fft-forward.txt";
const char*     _bak_file_name  = "fft-backward.txt";

bool            _time           = false;
int             _fft_size       = 8192;
int             _count          = 1000;
float           _mean           = 0.5;
float           _std            = 0.2;
float           _invert         = false;
bool            _use_periodic   = false;

vector<float>  _data;
vector<float>  _output;

void allocate() {
    for (int i = 0; i < _fft_size; ++i) {
        _data.push_back(0.0);
        _output.push_back(0.0);
    }    
}

void randomize() {
    
    std::default_random_engine       generator(std::random_device{}());
    std::normal_distribution<float> distribution(_mean, _std);
    
    for (int i = 0; i < _fft_size; ++i) {
        _data[i] = distribution(generator);
    }
}

void periodic() {
    for (int i = 0; i < _fft_size; ++i) {
        float t = i * .002;
        float amp = sin(M_PI * t);
        amp += sin(2 * M_PI * t);
        amp += sin(3 * M_PI * t); 
        _data[i] = amp;
    }
}

void populate() {
    if (_use_periodic)
        periodic();
    else
        randomize();
}

void copy(vector<float>& src, vector<float>& dst) {
    for (int i = 0; i < src.size(); ++i) {
        dst.push_back(src[i]);
    }
}

void dump_fft(String label, vector<float>& data) {
    
    cout << label << " size " << data.size() << endl;
    for (int i = 0; i < 48 ; ++i) {
        cout << data[i] << endl;
        if (i % 8 == 0 && i != 0)
            cout << endl;
    }
    cout << endl;
}

void write_data(vector<float>& data, string filename) {
    ofstream ofs;
    ofs.open(filename);
    ofs.precision(10);

    for (int i = 0; i < data.size(); ++i) {
        ofs << data[i] << endl;
    }
    
    ofs.close();   
}

void write_data_ccs(vector<float>& data, string filename) {
    ofstream ofs;
    ofs.open(filename);
    ofs.precision(10);

    for (int i = 1; i < data.size() / 2; i+=2) {
        float amp = sqrt(pow(data[i], 2) + pow(data[i+1], 2));
        ofs << amp << endl;
    }
    
    ofs.close();
}

float signal_energy(vector<float>& input) {
    
    float si = 0;
    for (int i = 0; i < input.size(); ++i) {
        si += pow(input[i], 2);
    }
    return si;
}

float quant_err_energy(vector<float>& input, vector<float>& output)  {
    
    float qe = 0;
    for (int i = 0; i < input.size(); ++i) {
        qe += pow(input[i] - output[i], 2);
    }
    return qe;
}

float sqer(vector<float>& input, vector<float>& output) {
    float se = signal_energy(input);
    float qe = quant_err_energy(input, output);
    
    return 10 * log10(se / qe);
}

void write_fft() {

    vector<float> orig;

    populate();
    copy(_data, orig);
    write_data(_data, _data_file_name);
       
    dft(_data, _data, 0, _data.size());
    write_data_ccs(_data, _fft_file_name);
    
    dft(_data, _data, DFT_INVERSE | DFT_SCALE, _data.size());
    write_data(_data, _bak_file_name);

    cout << "Data size:  " << _fft_size << endl;
    cout << "Data type:  " << (_use_periodic ? "Periodic" : "Random") << endl;
    if (!_use_periodic) {
        cout << "Mean:       " << _mean << endl;
        cout << "Std Dev:    " << _std << endl;
    }
    cout << endl;
    cout << "SQER:       " << sqer(orig, _data) << endl;
}

void fft_sqer(nanoseconds& duration, float& error) {

    vector<float> orig;

    populate();
    copy(_data, orig);

    high_resolution_clock::time_point start = high_resolution_clock::now();

    dft(_data, _data, 0, _data.size());
    dft(_data, _data, DFT_INVERSE | DFT_SCALE, _data.size());
    
    high_resolution_clock::time_point finish = high_resolution_clock::now();

    duration = duration_cast<nanoseconds>(finish - start);
    error = sqer(orig, _data);
}

void fft(nanoseconds& duration) {

    populate();
    
    high_resolution_clock::time_point start = high_resolution_clock::now();
    
    dft(_data, _output, 0, _data.size());

    high_resolution_clock::time_point finish = high_resolution_clock::now();

    duration = duration_cast<nanoseconds>(finish - start);
}

void time_fft() {
    
    cerr << "0 %";
    cerr.flush();
    
    nanoseconds total_duration(0);
    float      total_sqer = 0;
    int         last_percent = -1;
    
    for (int i = 0; i < _count; ++i) {
        
        nanoseconds duration(0);
        float sqer = 0;
        if (_invert)
            fft_sqer(duration, sqer);
        else
            fft(duration);
        
        total_duration += duration;
        total_sqer += sqer;
        
        int percent = (int) ((float) i / (float) _count * 100.0);
        if (percent != last_percent) {
            cerr << "\r" << percent << " %    ";
            cerr.flush();
            last_percent = percent;
        }
    }
    
    float count    = _count * (_invert ? 2 : 1);
    float ave_dur  = total_duration.count() / count;
    float ave_sqer = total_sqer / _count;

    cout.precision(8);
    cerr << "\r100 % " << endl;
    cout << endl;
    cout << "Iterations: " << _count << endl;
    cout << "Data size:  " << _fft_size << endl;
    cout << "Data type:  " << (_use_periodic ? "Periodic" : "Random") << endl;
    if (!_use_periodic) {
        cout << "Mean:       " << _mean << endl;
        cout << "Std Dev:    " << _std << endl;
    }
    cout << endl;
    cout << "Time:       " << total_duration.count() << " ns" << endl;
    cout << "Average:    " << ave_dur << " ns (" << (ave_dur / 1000.0) << " μs)" << endl;
    
    if (_invert) { 
        cout << "SQER:       " << total_sqer << endl;
        cout << "Ave SQER:   " << ave_sqer << endl;    
    } 
}

int main(int ac, char* av[]) {

    try {
        
        po::options_description desc("Allowed options");
    
        desc.add_options()
        ("help,h",      "produce help message")
        ("time,t",      "Time the FFT operation")
        ("invert,i",    "Perform timings on both the  FFT and inverse FFT")
        ("count,c",     po::value<int>(), "set the number of timed loops to perform")
        ("size,s",      po::value<int>(), "Set the size of the data buffer [8192]")
        ("mean,m",      po::value<float>(), "Set the range of the random data [25.0]")
        ("deviation,d", po::value<float>(), "Set the minimum value of the random data [0.0]")
        ("periodic,p",  "Use periodic instead of random data");

        po::variables_map vm;
        po::store(po::parse_command_line(ac, av, desc), vm);
        po::notify(vm);    

        if (vm.count("help")) {
            cout << desc << "\n";
            return 1;
        }
        
        if (vm.count("time")) {
            _time = true;
        }
        
        if (vm.count("invert")) {
            _invert = true;
        }

        if (vm.count("count")) {
            _count = vm["count"].as<int>();
        }
    
        if (vm.count("size")) {
            _fft_size = vm["size"].as<int>();
        }
        
        if (vm.count("mean")) {
            _mean = vm["mean"].as<float>();
        }
        
        if (vm.count("deviation")) {
            _std = vm["deviation"].as<float>();
        }
        
        if (vm.count("periodic")) {
            _use_periodic = true;
        }

        allocate();
        if (_time)
            time_fft();
        else
            write_fft();

    } catch (exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    } catch (...) {
        cerr << "Unknown error" << endl;
        return 1;
    }

    return 0;
}
