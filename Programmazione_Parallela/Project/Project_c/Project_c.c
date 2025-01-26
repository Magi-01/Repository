#include "ppm.h"
#include "scene.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

int main() {
    const char *file_path = "./to_be_raytraced.txt";  // Use the correct file path for your input

    // Parse the PPM file and store the data in an image structure
    PPMImage image = parse_ppm(file_path);

    // Render the scene based on the parsed data
    render_scene(&image);

    // Save the rendered image to a file
    save_ppm_image("output.ppm", &image);

    // Free dynamically allocated memory for spheres

    return 0;
}
