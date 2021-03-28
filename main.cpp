#include <iostream>

#include "fuzzy_set.hpp"
//#include "fuzzy.hpp"

using namespace _FUZZY;


int main()
{
    Fuzzy<double> sd1(1.2, 2.5, 2.5);
    Fuzzy<double> sd2(1.2, 2.5, 2.5);

    auto sd  = (sd1 + sd2);
//    std::cout << std::boolalpha << sd << std::endl;
//    input_output( cout_ << _1 << _2 );
    return 0;
}
