#pragma once

#include <boost/mpl/bool.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/for_each.hpp>
#include <boost/mpl/or.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/vector_c.hpp>
#include <boost/proto/proto.hpp>

#include <boost/multi_array.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>

#include <type_traits>
#include <array>
#include <complex>
#include <random>
#include <ctime>
#include <chrono>
#include <thread>


namespace mpl = boost::mpl;
namespace proto = boost::proto;

using float50 = boost::multiprecision::cpp_bin_float_50;

static constexpr int NZ {36}; //Кол наблюдаемых точек

static constexpr int step = 4;
static constexpr int freq = 4;

static constexpr int length = step * freq;
static const double edg_cube = .3;

static constexpr int sampleCountX = 64;
static constexpr int sampleCountZ = 64;
static constexpr int heightMapGridStepX = 4;
static constexpr int heightMapGridStepZ = 4;
static const float sampleMin = -16.0f;
static const float sampleMax = 16.0f;

///Factorial

template<size_t N>
struct factorial {
    static constexpr size_t value = N*factorial<N - 1>::value;
};
template<>
struct factorial<0> {
    static constexpr size_t value = 1;
};

///
///Вычисление степени
///
namespace my {
template<class T, int N>
    struct helper
    {
        static constexpr T pow(const T x){
            return helper<T, N-1>::pow(x) * x;
        }
    };

    template<class T>
    struct helper<T, 1>
    {
        static constexpr T pow(const T x){
            return x;
        }
    };

    template<class T>
    struct helper<T, 0>
    {
        static constexpr T pow(const T x){
            return T(1);
        }
    };
    template<int N, class T>
    T constexpr pow(T const x)
    {
        return helper<T, N>::pow(x);
    }
}

// is vector for multiply vfnrbx on vector
template <typename F, size_t N, typename Vector>
struct is_vector : boost::mpl::false_ {};
template <typename F, size_t N>
struct is_vector<F, N, boost::array<F, N>> : boost::mpl::true_ {
    static constexpr size_t size = N;
};
template <typename F, size_t N>
struct is_vector<F, N, std::array<F, N>> : boost::mpl::true_ {
    static constexpr size_t size = N;
};


/*!
 * struct meta_loop
 */
template <size_t N, size_t I, class Closure>
typename std::enable_if_t<(I == N)> is_meta_loop(Closure&) {}

template <size_t N, size_t I, class Closure>
typename std::enable_if_t<(I < N)> is_meta_loop(Closure& closure) {
    closure.template apply<I>();
    is_meta_loop<N, I + 1>(closure);
}
template <size_t N, class Closure>
void meta_loop(Closure& closure) {
    is_meta_loop<N, 0>(closure);
}
template <size_t N, class Closure>
void meta_loopUV(Closure& closure) {
    is_meta_loop<N, 1>(closure);
}
template <size_t N, size_t K, class Closure>
void meta_loop_KN(Closure& closure) {
    is_meta_loop<N, K>(closure);
}
///++
///
/*!
 * struct meta_loop_inv
 */
template <int N, int I, class Closure>
typename std::enable_if_t<(I < 0)> is_meta_loop_inv(Closure&) {}

template <int N, int I, class Closure>
typename std::enable_if_t<(I >= 0)> is_meta_loop_inv(Closure& closure) {
    closure.template apply<I>();
    is_meta_loop_inv<0, I - 1>(closure);
}
template <int N, class Closure>
void meta_loop_inv(Closure& closure) {
    is_meta_loop_inv<0, N>(closure);
}

///++

/////Calculate Binom

template<size_t N, size_t K>
struct BC {
    static constexpr size_t value = factorial<N>::value / factorial<K>::value / factorial<N - K>::value;
};
/*!
 * struct abstract_sum
 */
template<class Closure>
struct abstract_sum_closures {
    typedef typename Closure::value_type value_type;
    abstract_sum_closures(Closure &closure) :  closure(closure), result(value_type(0)){}

    template<unsigned I>
    void apply(){
        result += closure.template value<I>();
    }
    Closure &closure;
    value_type result;
};

template<size_t N, class Closure>
typename Closure::value_type abstract_sums(Closure &closure) {
    abstract_sum_closures<Closure> my_closure(closure);
    meta_loop<N>(my_closure);
    return my_closure.result;
}

/*!
 * struct abstract_subtract
 */
template<class Closure>
struct abstract_subtract_closures {
    typedef typename Closure::value_type value_type;
    abstract_subtract_closures(Closure &closure) :  closure(closure), result(value_type()){}

    template<unsigned I>
    void apply(){
        result -= closure.template value<I>();
    }
    Closure &closure;
    value_type result;
};

template<unsigned N, class Closure>
typename Closure::value_type abstract_subtract(Closure &closure) {
    abstract_subtract_closures<Closure> my_closure(closure);
    meta_loop<N>(my_closure);
    return my_closure.result;
}
/*!
 * struct abstract_mult
 */
template<class Closure>
struct abstract_multiple_closures {
    using value_type = typename Closure::value_type;
    abstract_multiple_closures(Closure &closure) : closure(closure), result(value_type(1)){}
    template<size_t I>
    void apply(){
        result *= closure.template value<I>();
    }
    Closure &closure;
    value_type result;
};
template<size_t K, class Closure>
typename Closure::value_type abstract_multiple(Closure &closure) {
    abstract_multiple_closures<Closure> my_closure(closure);
    meta_loop<K>(my_closure);
    return my_closure.result;
}

/*!
 * struct abstract_divide
 */
template<class Closure>
struct abstract_divide_closures {
    typedef typename Closure::value_type value_type;
    abstract_divide_closures(Closure &closure) :  closure(closure), result(value_type(1)){}

    template<unsigned I>
    void apply(){
        result /= closure.template value<I>();
    }
    Closure &closure;
    value_type result;
};

template<unsigned N, class Closure>
typename Closure::value_type abstract_divide(Closure &closure) {
    abstract_subtract_closures<Closure> my_closure(closure);
    meta_loop<N>(my_closure);
    return my_closure.result;
}
///meta func
///
// is vector for multiply vfnrbx on vector
template <size_t N, typename Array>
struct is_arrayd : boost::mpl::false_ {};
template <size_t N>
struct is_arrayd<N, std::array<double, N>> : boost::mpl::true_ {
    static constexpr size_t size = N;
};

template<typename T> struct is_double : boost::mpl::false_ {};
template<> struct is_double<double> : boost::mpl::true_ {};

template<typename T> struct is_complexd : boost::mpl::false_ {};
template<> struct is_complexd<std::complex<double>> : boost::mpl::true_ {};

///Gen random lon lat
template<size_t N, typename T, typename Array, typename = std::enable_if_t<std::is_same_v<Array, std::array<T, N>>>>
inline void gen_rand_array(Array& A, T min, T max, int n = 0) {
    auto start = std::chrono::system_clock::now();
    std::this_thread::sleep_for(std::chrono::milliseconds(n));
    auto end = std::chrono::system_clock::now();
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::mt19937 gen{static_cast<std::uint32_t>(end_time)};
    std::uniform_real_distribution<> dist{min, max};
    for(size_t i = 0; i < N; ++i)
            A[i] = dist(gen);
}

#pragma once

#include <boost/multi_array.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/mpl/if.hpp>
#include <boost/mpl/or.hpp>
#include <boost/random.hpp>
#include <boost/any.hpp>
#include <boost/hana.hpp>
#include <boost/proto/proto.hpp>

#include <utility>
#include <tuple>

#define _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS

namespace mpl = boost::mpl;
namespace proto = boost::proto;



template<typename Matrix, class = void>
struct IsMatrix : mpl::false_ {};
template<typename Matrix>
struct IsMatrix<Matrix, std::is_void<decltype (std::declval<Matrix&>().is_Matrix())>> : mpl::true_  {};

template<typename T> struct is_int : boost::mpl::false_ {};
template<> struct is_int<int> : boost::mpl::true_ {};
template<> struct is_int<unsigned> : boost::mpl::true_ {};

template<typename T, typename Matrix, typename = boost::enable_if_t<(Matrix::dimension > 0)>>
inline void rand_spatial_matrix(Matrix& A, T min, T max) {
    std::time_t now = std::time(0);
    boost::random::mt19937 gen{static_cast<std::uint32_t>(now)};

        if constexpr(is_int<T>::value) {
            boost::random::uniform_int_distribution<> dist{min, max};
            for(size_t i = 0; i < A.size(0); ++i)
                for(size_t j = 0; j < A.size(1); ++j)
                    for(size_t k = 0; k < A.size(2); ++k)
                        A(i, j, k) = dist(gen);
        } else {
            boost::random::uniform_real_distribution<> dist{min, max};
            for(size_t i = 0; i < A.size(0); ++i)
                for(size_t j = 0; j < A.size(1); ++j)
                    for(size_t k = 0; k < A.size(2); ++k)
                        A(i, j, k) = dist(gen);
        }

}



namespace _spatial {

template<typename T>
struct matrix_3 : boost::multi_array<T, 3> {
    static constexpr size_t dimension = 3;
    typedef boost::multi_array<T, 3> array_type;
    typedef boost::multi_array_types::index_range range;
    typedef typename array_type::template array_view<1>::type sub_matrix_view1D;
    typedef typename array_type::template array_view<2>::type sub_matrix_view2D;
    typedef typename array_type::template array_view<3>::type sub_matrix_view3D;
    typedef typename array_type::index index_type;
    typedef T value_type;
    array_type MTRX;
public:
    constexpr matrix_3(const std::array<index_type, 3>& shape) : MTRX(shape) {}
    constexpr matrix_3(size_t N, size_t M, size_t K) : MTRX({boost::extents[N][M][K]}) {}

    size_t size(size_t i) const {
        BOOST_ASSERT(i < 3);
        return MTRX.shape()[i];
    }
    constexpr void is_Matrix() {}

    constexpr T& operator () (size_t i, size_t j, size_t k) {
        return MTRX[i][j][k];
    }
    constexpr T const& operator () (size_t i, size_t j, size_t k) const {
        return MTRX[i][j][k];
    }

};

struct matrixGrammar : proto::or_<
    proto::terminal< matrix_3<proto::_> >,
    proto::plus< matrixGrammar, matrixGrammar>,
    proto::minus< matrixGrammar, matrixGrammar>,
    proto::negate< matrixGrammar>,
    proto::less_equal< matrixGrammar, matrixGrammar>,
    proto::greater_equal< matrixGrammar, matrixGrammar>,
    proto::less< matrixGrammar, matrixGrammar>,
    proto::greater< matrixGrammar, matrixGrammar>,
    proto::not_equal_to< matrixGrammar, matrixGrammar>,
    proto::equal_to< matrixGrammar, matrixGrammar>
> {};

template<typename Expr> struct Matrix_expr;
struct matrixDomain
    : proto::domain<proto::generator<Matrix_expr>, matrixGrammar> {};

template<typename Size = size_t>
struct SubscriptCntxt {

        Size i, j, k;
        SubscriptCntxt(Size i, Size j, Size k) : i(i), j(j), k(k) {}

        template<typename Expr, typename Tag = typename Expr::proto_tag>
        struct eval: proto::default_eval<Expr, SubscriptCntxt>{};

        template<typename Expr>
        struct eval<Expr, proto::tag::terminal> {
            typedef typename proto::result_of::value<Expr>::type::value_type result_type;
            result_type operator()( Expr const& expr, SubscriptCntxt& ctx ) const {
                return proto::value( expr)(ctx.i, ctx.j, ctx.k);
            }
        };

};
struct SizeCtx {
    SizeCtx(size_t Ni, size_t Nj, size_t Nk)
      : NI(Ni), NJ(Nj), NK(Nk){}
    template<typename Expr, typename EnableIf = void>
    struct eval : proto::null_eval<Expr, SizeCtx const> {};

    template<typename Expr>
    struct eval<Expr, typename boost::enable_if<
            proto::matches<Expr, proto::terminal<matrix_3<proto::_> > >
        >::type
    >
    {
        typedef void result_type;

        result_type operator ()(Expr &expr, SizeCtx const &ctx) const
        {
            if(ctx.NI != proto::value(expr).size(0)) {
                throw std::runtime_error("Матрицы не совпадают в размерности по индексу i");
            } else if (ctx.NJ != proto::value(expr).size(1)) {
                throw std::runtime_error("Матрицы не совпадают в размерности по индексу j");
            } else if (ctx.NK != proto::value(expr).size(2)) {
                throw std::runtime_error("Матрицы не совпадают в размерности по индексу k");
            }
        }
    };

    size_t NI;
    size_t NJ;
    size_t NK;
};

template<typename Expr>
struct Matrix_expr: proto::extends<Expr, Matrix_expr<Expr>, matrixDomain> {
    Matrix_expr( Expr const& expr= Expr() ): Matrix_expr::proto_extends( expr){}

    template< typename Size>
    typename proto::result_of::eval< Expr, SubscriptCntxt<Size> >::type
    operator()( Size i, Size j, Size k) const{
        SubscriptCntxt<Size> ctx(i, j, k);
        return proto::eval(*this, ctx);
    }
};

template< typename T >
struct Matrix3D : Matrix_expr< typename proto::terminal< matrix_3<T> >::type> {
private:
    typedef typename proto::terminal< matrix_3<T> >::type expr_type;
    typedef typename matrix_3<T>::array_type array_type;
    typedef typename matrix_3<T>::index_type index_type;
    typedef boost::multi_array_types::index_range range;
    typedef typename matrix_3<T>::sub_matrix_view1D  sub_matrix_view1D;
    const std::array<index_type, 3>& shape_;
public:
    Matrix3D(const std::array<index_type, 3>& shape) :
        Matrix_expr<expr_type>(expr_type::make( matrix_3<T>(shape) ) ), shape_(shape){}
    void Random(T min, T max) {
        rand_spatial_matrix(proto::value(*this), min, max);
    }
    size_t size_(size_t i) const {
        return proto::value(*this).size(i - 1);
    }
    template< typename Expr>
    Matrix3D& operator = (Expr const& expr) {
        SizeCtx const sizes(size_(1),
                            size_(2),
                            size_(3));

        proto::eval(proto::as_expr<matrixDomain>(expr), sizes);
        for(size_t i = 0; i < size_(1); ++i) {
            for(size_t j = 0; j < size_(2); ++j) {
                for(size_t k = 0; k < size_(3); ++k) {
                    proto::value(*this)(i, j, k) = expr(i, j, k);
                }
            }
        }
        return *this;
    }
    template< typename Expr>
    Matrix3D& operator += (Expr const& expr) {
        SizeCtx const sizes(size_(1),
                            size_(2),
                            size_(3));

        proto::eval(proto::as_expr<matrixDomain>(expr), sizes);
        for(size_t i = 0; i < size_(1); ++i) {
            for(size_t j = 0; j < size_(2); ++j) {
                for(size_t k = 0; k < size_(3); ++k) {
                    proto::value(*this)(i, j, k) += expr(i, j, k);
                }
            }
        }
        return *this;
    }
    Matrix3D operator + (const T val) {
        Matrix3D matrix(shape_);
        for(size_t i = 0; i < size_(1); ++i) {
            for(size_t j = 0; j < size_(2); ++j) {
                for(size_t k = 0; k < size_(3); ++k) {
                    proto::value(matrix)(i, j, k) = proto::value(*this)(i, j, k) + val;
                }
            }
        }
        return matrix;
    }
    Matrix3D operator * (const T val) {
        Matrix3D matrix(shape_);
        for(size_t i = 0; i < size_(1); ++i) {
            for(size_t j = 0; j < size_(2); ++j) {
                for(size_t k = 0; k < size_(3); ++k) {
                    proto::value(matrix)(i, j, k) = proto::value(*this)(i, j, k) * val;
                }
            }
        }
        return matrix;
    }
    Matrix3D operator / (const T val) {
        Matrix3D matrix(shape_);
        for(size_t i = 0; i < size_(1); ++i) {
            for(size_t j = 0; j < size_(2); ++j) {
                for(size_t k = 0; k < size_(3); ++k) {
                    proto::value(matrix)(i, j, k) = proto::value(*this)(i, j, k) / val;
                }
            }
        }
        return matrix;
    }
    friend std::ostream& operator << (std::ostream& os, const Matrix3D& A){
        for(const auto& x : proto::value(A).MTRX) {
            for(const auto& y : x) {
                for(const auto& z : y) os << z << "\t";
                os << std::endl;
            } os << std::endl;
        } os << std::endl;
        return os;
    }
    auto transversal_section(char index) {
        assert ((index == 'i' )|| (index == 'j') || (index == 'k'));
        typename array_type::index_gen indices;
        std::vector<sub_matrix_view1D> transversal_vector;
        std::vector<std::array<size_t, 3>> index_vector {};
        if(index == 'i'){
            for (index_type j = 0; j != size_(2); ++j) {
                for (index_type k = j; k != size_(3); ++k) {
                    if((j + k) == size_(1)) {
                        transversal_vector.push_back({proto::value(*this).MTRX[ indices[range(0, size_(1))][j][k] ]});
                        transversal_vector.push_back({proto::value(*this).MTRX[ indices[range(0, size_(1))][k][j] ]});
                        for (index_type i = 0; i != size_(1); ++i) {
                            index_vector.push_back({i, j, k});
                        }
                        for (index_type i = 0; i != size_(1); ++i) {
                            index_vector.push_back({i, k, j});
                        }
                    } else if(j == k && (j + k) < 1/*(size_(1) - k)*/) {
                        transversal_vector.push_back({proto::value(*this).MTRX[ indices[range(0, size_(1))][j][k] ]});
                        for (index_type i = 0; i != size_(1); ++i) {
                            index_vector.push_back({i, j, k});
                        }
                    }
                }
            }
        }
        if(index == 'j'){
            for (index_type i = 0; i != size_(1); ++i) {
                for (index_type k = i; k != size_(3); ++k) {
                    if((i + k) == size_(1)) {
                        transversal_vector.push_back({proto::value(*this).MTRX[ indices[i][range(0, size_(2))][k] ]});
                        transversal_vector.push_back({proto::value(*this).MTRX[ indices[k][range(0, size_(2))][i] ]});
                        for (index_type j = 0; j != size_(2); ++j) {
                            index_vector.push_back({i, j, k});
                        }
                        for (index_type j = 0; j != size_(2); ++j) {
                            index_vector.push_back({k, j, i});
                        }
                    } else if(i == k && (i + k) < 1/*(size_(1) - k)*/) {
                        transversal_vector.push_back({proto::value(*this).MTRX[ indices[i][range(0, size_(2))][k] ]});
                        for (index_type j = 0; j != size_(2); ++j) {
                            index_vector.push_back({i, j, k});
                        }
                    }
                }
            }
        }
        if(index == 'k'){
            for (index_type i = 0; i != size_(1); ++i) {
                for (index_type j = i; j != size_(2); ++j) {
                    if((i + j) == size_(1)) {
                        transversal_vector.push_back({proto::value(*this).MTRX[ indices[i][j][range(0, size_(3))] ]});
                        transversal_vector.push_back({proto::value(*this).MTRX[ indices[j][i][range(0, size_(3))] ]});
                        for (index_type k = 0; k != size_(3); ++k) {
                            index_vector.push_back({i, j, k});
                        }
                        for (index_type k = 0; k != size_(3); ++k) {
                            index_vector.push_back({j, i, k});
                        }
                    } else if(i == j && (i + j) < 1/*(size_(1) - k)*/) {
                        transversal_vector.push_back({proto::value(*this).MTRX[ indices[i][j][range(0, size_(3))] ]});
                        for (index_type k = 0; k != size_(3); ++k) {
                            index_vector.push_back({i, j, k});
                        }
                    }
                }
            }
        }
       return std::make_pair(transversal_vector, index_vector);
    }
};


}


