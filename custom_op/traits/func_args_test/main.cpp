#include <iostream>
#include <string>
#include <vector>

#include "FunctionTraits.h"

using namespace std;

class Node
{
    public:
    int first;
    int second;
};

template <typename traits, std::size_t... INDEX>
typename traits::ArgsNameTuple
dereference_vec_impl( std::index_sequence<INDEX...> )
{
    return { typeid(typename traits::template arg<INDEX>::type).name()... };
}

template <typename traits>
typename traits::ArgsNameTuple
GetNameList( )
{
    using Indices = std::make_index_sequence<traits::arity>;
    
    return dereference_vec_impl<traits>( Indices{} );
}

template<typename func_t>
void test_func(func_t&& op)
{
    //auto out = op( a, b, c);

    //cerr << out[0] << endl;

    using traits = function_traits<func_t>;
    cerr << "args num " << traits::arity << endl;
    cerr << "return type" << typeid( typename traits::result_type ).name() << endl;
    
    auto res = GetNameList<traits>();
    cerr << "name list num " << res.size() << endl; 

    for ( size_t i = 0; i < res.size(); ++i )
    {
        cerr << res[i] << endl;
    }
}

vector<int> func( int a, int b, Node c)
{
    return { a + b + c.first * c.second }; 
}

int main()
{
    
    //using traits = function_traits<func>;

    //cerr << "args num " << traits::arity << endl;

    int a = 1;
    int b = 2;
    Node n;
    n.first = 2;
    n.second = 5;

    auto c = func( a, b, n);

    cerr << "c is " << c[0] << endl;

    test_func(func);

    cerr << typeid(n).name() << endl;

}
