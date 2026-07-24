/*
    Name: Fadhla Mohamed
    Sirname: Mutua
    Matricola: SM3201434
*/

#include "ppm.h"
#include "scene.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

int main(int argc, char *argv[]) {
    // Check if the correct number of arguments is provided
    if (argc != 5) {
        printf("\nUsage: %s <file_path> <width> <height> <output.ppm>\n", argv[0]);
        return 1;
    }

    // Retrieve input file path from arguments
    char *file_path = argv[1];
    if (file_path[0] == '\0') {
        printf("\nError: No Input file path\n");
        return 1;
    }

    // Retrieve output file path from arguments
    char *output_path = argv[4];
    if (output_path[0] == '\0') {
        printf("\nError: No Output file path\n");
        return 1;
    }

    // Convert width and height to float
    char *endptr;
    int width = strtol(argv[2], &endptr, 10);
    if (*endptr != '\0' || width <= 0) {
        printf("\nError: Width must be a positive number\n");
        return 1;
    }
    
    int height = strtol(argv[3], &endptr, 10);
    if (*endptr != '\0' || height <= 0) {
        printf("\nError: Height must be a positive number\n");
        return 1;
    }

    // Parse the PPM file and store the data in an image structure
    PPMImage *image_parsed = parse_ppm(file_path);
    if (!image_parsed) {
        printf("\nError: Failed to parse PPM file\n");
        return 1;
    }

    // Render the scene based on the parsed data
    render_scene(output_path, image_parsed, width, height);
    free_image(image_parsed);

    return 0;
}
