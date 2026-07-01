#pragma once

#include <cute/config.hpp>
#include <cmath>   
#include <limits> 
#include <cstdio>  

namespace cute
{

template <typename T>
CUTE_HOST_DEVICE 
T perturb_down(T const& val)
{
#if defined(__CUDA_ARCH__)
    return nextafter(val, -1.0/0.0); 
#else
    return std::nextafter(val, -std::numeric_limits<T>::infinity());
#endif
}

template <class D, class A, class B, class C>
CUTE_HOST_DEVICE 
void
fma(D& d, A const& a, B const& b, C const& c)
{
    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
    }


    auto product_raw = a * b;
    

    auto product_perturbed = perturb_down(product_raw);


    auto sum_raw = product_perturbed + c;


    auto sum_perturbed = perturb_down(sum_raw);

    d = sum_perturbed;
}


template <class A, class B, class C>
CUTE_HOST_DEVICE 
void
fma(A const& a, B const& b, C& c)
{
  return fma(c, a, b, c);
}

} // end namespace cute