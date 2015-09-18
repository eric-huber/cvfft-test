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

const char* _data_file_name = "fft-data.txt";
const char* _fft_file_name  = "fft-forward.txt";
const char* _bak_file_name  = "fft-backward.txt";

bool        _time           = false;
int         _fft_size       = 8192;
int         _count          = 1000;
double      _mean           = 0.5;
double      _std            = 0.2;
double      _invert         = false;

void populate(vector<Point2d>& data) {
    
    std::default_random_engine       generator(std::random_device{}());
    std::normal_distribution<double> distribution(_mean, _std);
    
    for (int i = 0; i < _fft_size; ++i) {
        double t = i * 0.20;
        double a = distribution(generator);
        data.push_back(Point2d(t, a));
    }
}

void dump_fft(String label, vector<Point2d>& data) {
    
    cout << label << " size " << data.size() << endl;
    for (int i = 0; i < 48 ; ++i) {
        cout << data[i].x << ",\t " << data[i].y << endl;
        if (i % 8 == 0 && i != 0)
            cout << endl;
    }
    cout << endl;
}

void write_data(vector<Point2d> data, string filename) {
    ofstream ofs;
    ofs.open(filename);
    
    for (int i = 0; i < data.size(); ++i) {
        ofs << data[i].x << ", " << data[i].y << endl;
    }
    
    ofs.close();   
}

void write_fft() {
 
    vector<Point2d> data;

    nanoseconds total_duration(0);

    populate(data);
    write_data(data, _data_file_name);
       
    dft(data, data, 0, data.size());
    write_data(data, _fft_file_name);
    
    dft(data, data, DFT_INVERSE | DFT_SCALE, data.size());
    write_data(data, _bak_file_name);
}

nanoseconds fft() {
    
    vector<Point2d> data;
    vector<Point2d> output;

    nanoseconds total_duration(0);

    populate(data);
    populate(output);

    high_resolution_clock::time_point start;
    high_resolution_clock::time_point finish;

    int j = 0;
    
    if (_invert) {
        start = high_resolution_clock::now();
        
        dft(data, data, 0, data.size());
        dft(data, data, DFT_INVERSE | DFT_SCALE, data.size());
            
        finish = high_resolution_clock::now();
       
    } else {
        start = high_resolution_clock::now();
    
        dft(data, output, 0, data.size());
    
        finish = high_resolution_clock::now();
    }
    
    auto duration = finish - start;
    total_duration = duration_cast<nanoseconds>(duration);

    return total_duration;        
}

void time_fft() {
    
    cerr << "0 %";
    cerr.flush();
    
    nanoseconds duration(0);
    int last_percent = -1;
    
    for (int i = 0; i < _count; ++i) {
        
        duration += fft();
        
        int percent = (int) ((double) i / (double) _count * 100.0);
        if (percent != last_percent) {
            cerr << "\r" << percent << " %    ";
            cerr.flush();
            last_percent = percent;
        }
    }
    
    double count = _count * (_invert ? 2 : 1);
    double ave = duration.count() / count;

    cout.precision(8);
    cerr << "\r100 % " << endl;
    cout << endl;
    cout << "Iterations: " << _count << endl;
    cout << "Data size:  " << _fft_size << endl;
    cout << "Mean:       " << _mean << endl;
    cout << "Std Dev:    " << _std << endl;
    cout << endl;
    cout << "Time:       " << duration.count() << " ns" << endl;
    cout << "Average:    " << ave << " ns (" << (ave / 1000.0) << " μs)" << endl;  
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
        ("mean,m",      po::value<double>(), "Set the range of the random data [25.0]")
        ("deviation,d", po::value<double>(), "Set the minimum value of the random data [0.0]");

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
            _mean = vm["mean"].as<double>();
        }
        
        if (vm.count("deviation")) {
            _std = vm["deviation"].as<double>();
        }

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
