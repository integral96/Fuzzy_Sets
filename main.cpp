#include <iostream>

#include "fuzzy_set.hpp"
//#include "fuzzy.hpp"

int main()
{
    Fuzzy_set<double, int> sd1{1.2, 2};
    Fuzzy_set<double, int> sd2{2.2, 2};

    auto sd  = (sd1);
    std::cout << std::boolalpha << sd.template at<0>() << std::endl;
//    input_output( cout_ << _1 << _2 );
    return 0;
}
