#include "pch.h"

int main(int argc, char const **argv) {

    fmt::print("Main engine started.\n");

    gii::Gii& ui = gii::Gii::get_instance();
    ui.initilize();
    ui.present();

    return 0;
}
