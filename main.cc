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
bool        _ave_sqer       = false;
int         _fft_size       = 8192;
int         _count          = 1000;
double      _mean           = 0.5;
double      _std            = 0.2;
double      _invert         = false;

vector<Point2d>     _data;
vector<Point2d>     _output;

void allocate() {
    for (int i = 0; i < _fft_size; ++i) {
        _data.push_back(Point2d(0.0, 0.0));
        _output.push_back(Point2d(0.0, 0.0));
    }    
}

void randomize() {
    
    std::default_random_engine       generator(std::random_device{}());
    std::normal_distribution<double> distribution(_mean, _std);
    
    for (int i = 0; i < _fft_size; ++i) {
        double t = i * 0.20;
        double a = distribution(generator);
        _data[i].x = t;
        _data[i].y = a;
    }
}

void copy(vector<Point2d>& src, vector<Point2d>& dst) {
    for (int i = 0; i < src.size(); ++i) {
        dst.push_back(Point2d(src[i].x, src[i].y));
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

void write_data(vector<Point2d>& data, string filename) {
    ofstream ofs;
    ofs.open(filename);
    
    for (int i = 0; i < data.size(); ++i) {
        ofs << data[i].x << ", " << data[i].y << endl;
    }
    
    ofs.close();   
}

double signal_energy(vector<Point2d>& input) {
    
    double si = 0;
    for (int i = 0; i < input.size(); ++i) {
        si += pow(input[i].y, 2);
    }
    return si;
}

double quant_err_energy(vector<Point2d>& input, vector<Point2d>& output)  {
    
    double qe = 0;
    for (int i = 0; i < input.size(); ++i) {
        qe += pow(input[i].y - output[i].y, 2);
    }
    return qe;
}

double sqer(vector<Point2d>& input, vector<Point2d>& output) {
    double se = signal_energy(input);
    double qe = quant_err_energy(input, output);
    
    return 10 * log10(se / qe);
}

void write_fft() {

    vector<Point2d> orig;

    randomize();
    copy(_data, orig);
    write_data(_data, _data_file_name);
       
    dft(_data, _data, 0, _data.size());
    write_data(_data, _fft_file_name);
    
    dft(_data, _data, DFT_INVERSE | DFT_SCALE, _data.size());
    write_data(_data, _bak_file_name);

    cout << "Mean:       " << _mean << endl;
    cout << "Std Dev:    " << _std << endl;
    cout << endl;
    cout << "SQER:       " << sqer(orig, _data) << endl;
}

double fft_sqer() {

    vector<Point2d> orig;

    randomize();
    copy(_data, orig);
       
    dft(_data, _data, 0, _data.size());
    dft(_data, _data, DFT_INVERSE | DFT_SCALE, _data.size());

    return sqer(orig, _data);
}

void ave_sqr() {

    cerr << "0 %";
    cerr.flush();
    
    double sqer = 0;
    int last_percent = -1;
    
    for (int i = 0; i < _count; ++i) {
        
        sqer += fft_sqer();
        
        int percent = (int) ((double) i / (double) _count * 100.0);
        if (percent != last_percent) {
            cerr << "\r" << percent << " %    ";
            cerr.flush();
            last_percent = percent;
        }
    }
    
    sqer /= _count;

    cout << "Iterations: " << _count << endl;
    cout << "Mean:       " << _mean << endl;
    cout << "Std Dev:    " << _std << endl;
    cout << endl;
    cout << "Ave SQER:   " << sqer << endl;    
}

nanoseconds fft() {
    nanoseconds total_duration(0);
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point finish;

    randomize();
    
    if (_invert) {
        start = high_resolution_clock::now();
        
        dft(_data, _data, 0, _data.size());
        dft(_data, _data, DFT_INVERSE | DFT_SCALE, _data.size());
            
        finish = high_resolution_clock::now();
       
    } else {
        start = high_resolution_clock::now();
    
        dft(_data, _output, 0, _data.size());
    
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
        ("ave,a",       "Ave SQER")
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
        
        if (vm.count("ave")) {
            _ave_sqer = true;
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

        allocate();
        if (_time)
            time_fft();
        else if (_ave_sqer) 
            ave_sqr();
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
