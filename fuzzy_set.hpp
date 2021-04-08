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



template<typename ...Args>
struct Fuzzy_set;




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


template<size_t I>
struct Fuzzy_set_context {

    Fuzzy_set_context()  {}


    template<typename Expr, typename Tag = typename Expr::proto_tag>
    struct eval : proto::default_eval<Expr, Fuzzy_set_context>  {};


    template<typename Expr>
    struct eval<Expr, proto::tag::terminal>
    {
        typedef typename fusion::result_of::at_c<typename proto::result_of::value<Expr>::type, I>::type result_type;
        result_type operator ()( Expr const & expr, Fuzzy_set_context & ) const
        {
//            std::remove_const_t<decltype (proto::value(expr))> tmp;
            return fusion::at_c<mpl::int_<I>>(proto::value(expr)) ;
        }
    };
};


template<typename Expr>
struct Fuzzy_setExpr : proto::extends<Expr, Fuzzy_setExpr<Expr>, Fuzzy_set_Domain> {

    explicit Fuzzy_setExpr( Expr const & expr = Expr() )
          : Fuzzy_setExpr::proto_extends( expr ) {}

    template< size_t I>
//    typename proto::result_of::eval< Expr, Fuzzy_set_context<I> >::type
    typename fusion::result_of::at_c<typename proto::result_of::eval< Expr, Fuzzy_set_context<I> >::type, I>::type
    at( ) const
    {
        Fuzzy_set_context<I> const ctx;
        return proto::eval(*this, ctx);
    }
//    typename fusion::result_of::as_vector<
//                typename proto::result_of::eval< Expr, FuzzyGroup(Expr) >::type>::type
//    get() const {
//        return fusion::as_vector(FuzzyGroup()(*this));
//    }
};

template< typename ...Args >
struct Fuzzy_set
  : Fuzzy_setExpr< typename proto::terminal< fusion::vector<Args...> >::type >
{
    typedef typename proto::terminal< fusion::vector<Args...> >::type expr_type;

    Fuzzy_set(Args&& ... args)
        : Fuzzy_setExpr<expr_type>( expr_type::make( fusion::vector<Args...>{std::forward<Args>(args)...} ) )
    {}

//    template< typename Expr >
//    Fuzzy_set &operator += (Expr const & expr)
//    {
//        size_t size = fusion::size(proto::value(*this));
//        for(std::size_t i = 0; i < size; ++i)
//        {
//            proto::value(*this)[i] += expr[i];
//        }
//        return *this;
//    }
    template< typename Expr >
    Fuzzy_set &operator = (Expr const & expr)
    {
        proto::value(*this).assign_sequence(proto::value(expr));
        return *this;
    }
};



////=====================================
