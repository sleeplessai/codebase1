#include "pch.h"

int main(int argc, char const **argv) {
	fmt::print("Main engine started.\n");

	ui::MainUI& ui = ui::MainUI::get_instance();
	ui.initilize();
	ui.present();

	return 0;
}
