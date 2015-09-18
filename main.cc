#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <boost/program_options.hpp>

#include <cstdlib>
#include <chrono>
#include <iostream>
 
using namespace std;
using namespace cv;
using namespace chrono;

namespace po = boost::program_options;

int     _fft_size       = 8192;
int     _count          = 1000;
int     _count_per_loop = 1000;
double  _mean           = 5.0;
double  _std            = 2.0;
double  _invert         = false;

void populate(vector<Point2d>& data) {
    
    std::default_random_engine       generator(std::random_device{}());
    std::normal_distribution<double> distribution(_mean, _std);
    
    for (int i = 0; i < _fft_size; ++i) {
        double time = i * 0.20;
        double amp = distribution(generator);
        data.push_back(Point2d(time, amp));
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

void time_fft() {

    vector<Point2d> data;
    vector<Point2d> output;

    cout.precision(2);
    cerr << "0 %";
    cerr.flush();

    nanoseconds total_duration(0);

    for (int i = 0; i < _count; ++i) {

        populate(data);
        if (0 == i)
            populate(output); // allocate space in output

        int j = 0;
        
        if (_invert) {
            high_resolution_clock::time_point start = high_resolution_clock::now();
            for (; j < _count_per_loop; ++j) {
                dft(data, data, 0, data.size());
                dft(data, data, DFT_INVERSE | DFT_SCALE, data.size());
            }        
            high_resolution_clock::time_point finish = high_resolution_clock::now();

            auto duration = finish - start;
            total_duration += duration_cast<nanoseconds>(duration);
            
        } else {
            high_resolution_clock::time_point start = high_resolution_clock::now();
            for (; j < _count_per_loop; ++j) {
                dft(data, output, 0, data.size());
            }        
            high_resolution_clock::time_point finish = high_resolution_clock::now();
         
            auto duration = finish - start;
            total_duration += duration_cast<nanoseconds>(duration);
        }        
       
        if (i % 10 == 0) {
            double percent = ((double) i / (double) _count * 100.0);
            cerr << "\r" << percent << " %    ";
            cerr.flush();
        }
    }
    
    double count = _count * _count_per_loop * (_invert ? 2 : 1);
    double ave = total_duration.count() / count;

    cout.precision(8);
    cerr << "\r100 % " << endl;
    cout << endl;
    cout << "Iterations: " << _count << endl;
    cout << "Per loop:   " << _count_per_loop << endl;
    cout << "Data size:  " << _fft_size << endl;
    cout << "Mean:       " << _mean << endl;
    cout << "Std Dev:    " << _std << endl;
    cout << endl;
    cout << "Time:       " << total_duration.count() << " ns" << endl;
    cout << "Average:    " << ave << " ns (" << (ave / 1000.0) << " μs)" << endl;  
}

int main(int ac, char* av[]) {

    try {
        
        po::options_description desc("Allowed options");
    
        desc.add_options()
        ("help,h",      "produce help message")
        ("invert,i",    "Perform an FFT, then an inverse FFT on the same data")
        ("count,c",     po::value<int>(), "set the number of timed loops to perform")
        ("loop,l",      po::value<int>(), "Set the number of FFT interations per loop")
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
        
        if (vm.count("invert")) {
            _invert = true;
        }

        if (vm.count("count")) {
            _count = vm["count"].as<int>();
        }
    
        if (vm.count("loop")) {
            _count_per_loop = vm["loop"].as<int>();
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

        time_fft();

    } catch (exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    } catch (...) {
        cerr << "Unknown error" << endl;
        return 1;
    }

    return 0;
}
