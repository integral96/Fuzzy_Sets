#pragma once

#include "base_class.hpp"
#include "fuzzy.hpp"

#include <boost/proto/proto.hpp>
#include <boost/array.hpp>
#include <boost/type_traits.hpp>
#include <boost/proto/operators.hpp>
#include <boost/fusion/container.hpp>
#include <boost/range/adaptor/strided.hpp>

#include <vector>
#include <type_traits>

namespace fusion = boost::fusion;


static constexpr size_t N1 = 10;
static constexpr size_t N2 = 10;
static constexpr size_t N3 = 10;

template<typename I> struct placeholder : I {};



template<typename Expr>
class Fuzzy_set;




/// ========================================
struct Fuzzy_grammar;

struct Fuzzy_grammarCases {
    template<typename Tag>
    struct case_ : proto::not_<proto::_> {};
};
template<>
struct Fuzzy_grammarCases::case_<proto::tag::terminal> :
    proto::or_<
                proto::terminal<float50>,
                proto::terminal<float50>,
                proto::terminal<float50>
                >{};
template<>
struct Fuzzy_grammarCases::case_<proto::tag::unary_plus> :
    proto::unary_plus<Fuzzy_grammar> {};
template<>
struct Fuzzy_grammarCases::case_<proto::tag::negate> :
    proto::negate<Fuzzy_grammar> {};
template<>
struct Fuzzy_grammarCases::case_<proto::tag::complement> :
    proto::complement<Fuzzy_grammar> {};
template<>
struct Fuzzy_grammarCases::case_<proto::tag::plus> :
    proto::plus<Fuzzy_grammar, Fuzzy_grammar> {};
template<>
struct Fuzzy_grammarCases::case_<proto::tag::minus> :
    proto::minus<Fuzzy_grammar, Fuzzy_grammar> {};
template<>
struct Fuzzy_grammarCases::case_<proto::tag::multiplies> :
    proto::multiplies<Fuzzy_grammar, Fuzzy_grammar> {};
template<>
struct Fuzzy_grammarCases::case_<proto::tag::logical_not> :
    proto::logical_not<Fuzzy_grammar> {};
template<>
struct Fuzzy_grammarCases::case_<proto::tag::logical_and> :
    proto::logical_and<Fuzzy_grammar, Fuzzy_grammar> {};
template<>
struct Fuzzy_grammarCases::case_<proto::tag::logical_or> :
    proto::logical_or<Fuzzy_grammar, Fuzzy_grammar> {};
template<>
struct Fuzzy_grammarCases::case_<proto::tag::less> :
    proto::less<Fuzzy_grammar, Fuzzy_grammar> {};
template<>
struct Fuzzy_grammarCases::case_<proto::tag::greater> :
    proto::greater<Fuzzy_grammar, Fuzzy_grammar> {};
template<>
struct Fuzzy_grammarCases::case_<proto::tag::less_equal> :
    proto::less_equal<Fuzzy_grammar, Fuzzy_grammar> {};
template<>
struct Fuzzy_grammarCases::case_<proto::tag::greater_equal> :
    proto::greater_equal<Fuzzy_grammar, Fuzzy_grammar> {};
template<>
struct Fuzzy_grammarCases::case_<proto::tag::equal_to> :
    proto::equal_to<Fuzzy_grammar, Fuzzy_grammar> {};
template<>
struct Fuzzy_grammarCases::case_<proto::tag::not_equal_to> :
    proto::not_equal_to<Fuzzy_grammar, Fuzzy_grammar> {};


struct Fuzzy_grammar : proto::switch_<Fuzzy_grammarCases> {};
/// ========================================
///

template<typename Expr>
struct Fuzzy_setExpr;

struct Fuzzy_set_Domain : proto::domain<proto::generator<Fuzzy_setExpr>, Fuzzy_grammar> {};

struct Fuzzy_set_context
  : proto::callable_context< Fuzzy_set_context const >
{
    typedef Fuzzy<double> result_type;

    using vector_type = fusion::vector<result_type, result_type>;
    vector_type args;

    explicit Fuzzy_set_context(result_type d1 = result_type(), result_type d2 = result_type())
    {
        fusion::push_back(args, d1);
        fusion::push_back(args, d2);
    }


    template<size_t I>
    result_type operator()(proto::tag::terminal, const result_type&) const
    {
        return fusion::at<mpl::int_<I>>(args);
    }
    template<typename E1, typename E2>
    result_type operator()(proto::tag::plus, const E1& e1, const E2& e2) const {
        return proto::eval(e1, *this) + proto::eval(e2, *this);
    }
    template<typename E1, typename E2>
    result_type operator()(proto::tag::minus, const E1& e1, const E2& e2) const {
        return proto::eval(e1, *this) - proto::eval(e2, *this);
    }
    template<typename E1, typename E2>
    result_type operator()(proto::tag::multiplies, const E1& e1, const E2& e2) const {
        return proto::eval(e1, *this) * proto::eval(e2, *this);
    }
};

template<typename T>
struct IsVector : mpl::false_ {};
template<>
struct IsVector<Fuzzy<double>> : mpl::true_ {};

template<typename Expr>
struct Fuzzy_setExpr : proto::extends<Expr, Fuzzy_setExpr<Expr>, Fuzzy_set_Domain> {

    using type = typename proto::result_of::eval< Expr, Fuzzy_set_context>::type;
//    using type = Fuzzy<double>;

    Fuzzy_setExpr(const Expr& e = Expr())
        : proto::extends<Expr, Fuzzy_setExpr<Expr>, Fuzzy_set_Domain>(e) {
        auto const & ti = BOOST_CORE_TYPEID(type);
        std::cout << boost::core::demangled_name( ti ) << std::endl;
    }
    type operator ()(const type& a) const {
        Fuzzy_set_context const ctx(a);
        return proto::eval(*this, ctx);
    }

    type operator ()(const type& a, const type& b) const {
        Fuzzy_set_context const ctx(a, b);
        return proto::eval(*this, ctx);
    }
};

namespace _FUZZY {
    BOOST_PROTO_DEFINE_OPERATORS(IsVector, Fuzzy_set_Domain)
}





////=====================================
