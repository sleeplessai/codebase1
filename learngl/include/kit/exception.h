#pragma once

#include <stdexcept>
#include <string>


#ifndef LOCATE_RT_ERR
#define LOCATE_RT_ERR ("line " + std::to_string(__LINE__) + ", " + std::string(__FILE__))
#define THROW_RT_ERR throw std::runtime_error(LOCATE_RT_ERR);
#endif

