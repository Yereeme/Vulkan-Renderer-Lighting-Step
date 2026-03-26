
#include "RTG.hpp"

#include "Tutorial.hpp"

#include <iostream>

 
 
#include <stdexcept>
int main(int argc, char **argv) {
	//main wrapped in a try-catch so we can print some debug info about uncaught exceptions:
	try {
		 
	 
		//configure application:
		RTG::Configuration configuration;

		configuration.application_info = VkApplicationInfo{
			.pApplicationName = "Ayman A2 Materials",
			.applicationVersion = VK_MAKE_VERSION(0,0,0),
			.pEngineName = "Unknown",
			.engineVersion = VK_MAKE_VERSION(0,0,0),
			.apiVersion = VK_API_VERSION_1_3
		};
		bool print_usage = false;
		std::optional<Tutorial::CameraMode> forced_camera;

		try {
			configuration.parse(argc, argv);
			



		} catch (std::runtime_error &e) {
			std::cerr << "Failed to parse arguments:\n" << e.what() << std::endl;
			print_usage = true;
		}

	 ;

		//require --scene:
		if (configuration.scene_file.empty()) {
			std::cerr << "Missing required argument: --scene <file.s72>\n";
			print_usage = true;
		}

		if (print_usage) {
			std::cerr << "\nUsage:\n";
			std::cerr << "    bin/viewer.exe --scene <file.s72> [RTG options]\n\n";
			std::cerr << "RTG options:\n";
			RTG::Configuration::usage( [](const char *arg, const char *desc){ 
				std::cerr << "    " << arg << "\n        " << desc << std::endl;
			});
			return 1;
		}

		//loads vulkan library, creates surface, initializes helpers:
		RTG rtg(configuration);

		//initializes global (whole-life-of-application) resources:
		Tutorial application(rtg, configuration.scene_file, configuration.culling);//scene file gets passed into Tutorial
		 


		//main loop -- handles events, renders frames, etc:
		rtg.run(application);

		return 0;

	} catch (std::exception &e) {
		std::cerr << "Exception: " << e.what() << std::endl;
		return 1;
	}
}
