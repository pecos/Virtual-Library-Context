#include "foobar.h"
#include <iostream>

int foo_counter = 1;
int bar_counter = 1;

void foo() {
    std::cout << "foo() is called the " << foo_counter++ << " time." << std::endl;
    return;
}

void bar() {
    std::cout << "bar() is called the " << bar_counter++ << " time." << std::endl;
    return;
}