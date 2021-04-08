#pragma once

#include <base_func.hpp>

#include <limits>
#include <algorithm>

#include <boost/range/functions.hpp>
#include <boost/range/metafunctions.hpp>
#include <boost/range/as_literal.hpp>
#include <boost/range/join.hpp>
#include <boost/fusion/container.hpp>
#include <boost/fusion/sequence.hpp>
#include <boost/fusion/algorithm.hpp>

namespace fusion = boost::fusion;

template<typename To, typename From, typename... Rest>
constexpr bool is_converible_variadic_impl() {
    return
        std::is_convertible_v<From, To> &&
        is_converible_variadic_impl<To, Rest...>();
}
template<typename To, typename From>
constexpr bool is_converible_variadic_impl() {
    return std::is_convertible_v<From, To>;
}

template<typename T>
auto join(T&& arg) {
    return boost::make_iterator_range_n(&arg, 0);
}
template<typename T, typename... Rest>
auto join(T&& arg, Rest&& ... args) {
    return boost::join(
                        boost::make_iterator_range_n(&arg, sizeof... (Rest))
                    , join(std::forward<Rest>(args)...)
            );

}

template<typename T>
class Fuzzy
{
public:
    using value_type = T;

    Fuzzy(): _gamma(_defaultGamma), _grades{0}
    {
        setDomain(value_type(0), value_type(1));
    }
    Fuzzy(const value_type& xMin, const value_type& xMax) :
        _gamma(_defaultGamma), _grades{0}
    {
        setDomain(xMin, xMax);
    }
    Fuzzy(const Fuzzy& arg):
        _gamma(_defaultGamma), _grades{0}
    {
        *this = arg;
    }

    void setDomain(const value_type& xMin, const value_type& xMax) {
        _xMin = xMin; _xMax = xMax;
        _domainSize = xMax - xMin;
        if(_size != 1) _resolution = _domainSize/(_size - 1.0);
        else _resolution = value_type(1);
    }
    void fillGrade(const value_type& fill_val);
///===============================================

    template<typename ...Args, typename = std::enable_if_t<(std::is_convertible_v<value_type, Args> && ...)>>
    auto fillAll(Args&&... val) {
        return fusion::push_back(_grades, {std::forward<Args>(val)...});
    }
///====================================
    void normalize();

    void small();
    void large();
    void rectangle(const value_type& left, const value_type& right);
    void triangle(const value_type& left, const value_type& centr, const value_type& right);
    void trapezoid(const value_type& leftBot, const value_type& leftTop, const value_type& rightTop, const value_type& rightBot);

    template<class Lambda>
    void greterThen(const value_type& arg, const Lambda& lambda = Lambda());
    template<class Lambda>
    void lessThen(const value_type& arg, const Lambda& lambda = Lambda());
    template<class Lambda>
    void closeTo(const value_type& arg, const Lambda& lambda = Lambda());

    int supportMaxGradeIndex() const;
    int supportMinGradelndex() const;
    value_type supportMaxGrade() const;
    value_type supportMinGrade() const;
    value_type minGrade() const;
    value_type maxGrade() const;
    value_type centroid() const;

    value_type cardinality() const;
    value_type relativeCardinality() const;
    Fuzzy limit(const value_type& ceiling);
    Fuzzy alphaCut(const value_type& alpha) const;
    Fuzzy betaCut(const int beta) const;
    value_type entropy(const value_type& base = 2) const;
    value_type xMin() const { return _xMin; }
    value_type xMax() const { return _xMax; }
    value_type domainSize () const { return _domainSize; }
    value_type resolution () const { return _resolution; }

    int isSubSet(const Fuzzy& arg) const;

    template<size_t I>
    value_type at () {
        return fusion::at<mpl::int_<I>>(_grades);
    }
    value_type operator () (const value_type& x) const;
    value_type operator () (const int i, const int j) const;
    value_type operator () (const int i, const int j, const int k) const;

    Fuzzy& operator = (const Fuzzy& arg) {
        BOOST_STATIC_ASSERT(fusion::size(_grades) == fusion::size(arg._grades));
        _grades.assign_sequence(arg._grades);
        setDomain(arg._xMin, arg._xMax);
        return *this;
    }

    Fuzzy operator ! () const;
    Fuzzy operator && (const Fuzzy& arg) const;
    Fuzzy operator || (const Fuzzy& arg) const;

    template<class Closure>
    struct Fuzzy_plus {
        Fuzzy_plus(const Closure& A, const Closure& B, Closure& result) : A(A), B(B), result_(result) {}
        template<size_t I>
        void apply() const {
            value_type rslt =  fusion::at<mpl::int_<I>>(A._grades) + fusion::at<mpl::int_<I>>(B._grades);
            if(rslt < 1.0) fusion::push_back(result_._grades, rslt);
            else fusion::push_back(result_._grades, value_type(1));
        }
    private:
        const Closure& A;
        const Closure& B;
    public:
        Closure& result_;
    };
    template<size_t N>
    Fuzzy operator + (const Fuzzy& arg) const {
        Fuzzy result(arg);
        Fuzzy_plus<Fuzzy> closure(*this, arg, result);
        meta_loop<N>(closure);
        return closure.result_;
    }
    Fuzzy operator - (const Fuzzy& arg) const;
    Fuzzy operator % (const Fuzzy& arg) const;
    Fuzzy operator * (const Fuzzy& arg) const;
    Fuzzy operator & (const Fuzzy& arg) const;
    Fuzzy operator | (const Fuzzy& arg) const;
    Fuzzy operator > (const Fuzzy& arg) const;
    Fuzzy operator < (const Fuzzy& arg) const;
    Fuzzy operator == (const Fuzzy& arg) const;
    Fuzzy operator >= (const Fuzzy& arg) const;
    Fuzzy operator <= (const Fuzzy& arg) const;
    Fuzzy operator != (const Fuzzy& arg) const;

    value_type gamma() const {return _gamma; }
    void setGamma(const value_type& newGamma) { _gamma = newGamma; };

    static Fuzzy mean(const Fuzzy* const * const sets, const int nSets,
    const value_type* const weights = NULL);

    Fuzzy enhanceContrast() const;
    Fuzzy hedge(const value_type* hedgeExp) const;

public:

    friend std::ostream& operator << (std::ostream&, const Fuzzy&);
    friend std::istream& operator >> (std::istream&, Fuzzy&);


    static const value_type extremely = 4.0;
    static const value_type very = 2.0;
    static const value_type substantially = 1.5;
    static const value_type somewhat = 0.5;
    static const value_type slightly = 0.25;
    static const value_type vaguely = 0.03;

//    value_type* _grades;
    fusion::vector<T> _grades;
    int _size;
    value_type _gamma;

    value_type _xMin, _xMax, _resolution, _domainSize;
    void resize(int newSize) {
//        delete [] _grades;
//        _grades = new value_type[newSize];
//        _size = newSize;
    }
private:
    static const value_type _defaultGamma = 0.5;

    value_type linearlnterpolate(const value_type* x0,const value_type* y0,
        const value_type* x1,const value_type* y1,
        const value_type* x) const;

};

