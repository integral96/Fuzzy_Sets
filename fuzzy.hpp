#pragma once

#include <base_func.hpp>

#include <limits>
#include <algorithm>

#include <boost/range/functions.hpp>
#include <boost/range/metafunctions.hpp>
#include <boost/range/as_literal.hpp>
#include <boost/range/join.hpp>

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

    Fuzzy(): _gamma(_defaultGamma), _grades(nullptr), _size(0)
    {
        setDomain(value_type(0), value_type(1));
    }
    template<typename ...Args, typename = std::enable_if_t<(std::is_same_v<T, Args> && ...)>>
    Fuzzy(Args&& ... args) :
        _gamma(_defaultGamma), _grades(nullptr), _size(sizeof... (args))
    {
        value_type xMin{};
        value_type xMax{};
        for(auto& x : join(args...)) {
            if(x > xMin) {
                xMin = x;
            }
            if(x < xMax) xMax = x;
        }
        resize(_size);
        setDomain(xMin, xMax);
    }
    Fuzzy(const Fuzzy& arg):
        _gamma(_defaultGamma), _grades(nullptr), _size(arg._size)
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

    template<typename ...Args, typename = std::enable_if_t<(std::is_convertible_v<value_type, Args> && ...)>>
    void fillAll(Args&&... val) {
        for(int i = 0; i <sizeof... (Args); i++)
            std::invoke(_grades[i], std::forward<Args>(val)...);
    }
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

    value_type operator [] (const int i);
    value_type operator () (const value_type& x) const;
    value_type operator () (const int i, const int j) const;
    value_type operator () (const int i, const int j, const int k) const;

    Fuzzy& operator = (const Fuzzy& arg) {
        if(_size != arg._size) resize(arg._size);
        for(int i=0;i<_size;i++)
            _grades[i] = arg._grades[i];
        setDomain(arg._xMin,arg._xMax);
        return *this;
    }

    Fuzzy operator ! () const;
    Fuzzy operator && (const Fuzzy& arg) const;
    Fuzzy operator || (const Fuzzy& arg) const;
    Fuzzy operator + (const Fuzzy& arg) const {
        Fuzzy result(arg);
        for(int i=0;i<_size;i++) {
            value_type rslt = _grades[i] + arg._grades[i];
            if(rslt < 1.0) result._grades[i] = rslt;
            else result._grades[i] = value_type(1);
        }
        return result;
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

    value_type* _grades;
    int _size;
    value_type _gamma;

    value_type _xMin, _xMax, _resolution, _domainSize;
    void resize(int newSize) {
        delete [] _grades;
        _grades = new value_type[newSize];
        _size = newSize;
    }
private:
    static const value_type _defaultGamma = 0.5;

    value_type linearlnterpolate(const value_type* x0,const value_type* y0,
        const value_type* x1,const value_type* y1,
        const value_type* x) const;

};

