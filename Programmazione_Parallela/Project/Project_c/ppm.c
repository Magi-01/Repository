// you can use  __attribute__((packed))

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "error_handling.h"

// Define the PPMImage structure with packed attribute
typedef struct __attribute__((packed)) {
    char magic_number[2]; // PPM format identifier, e.g., "P6"
    uint32_t width;       // Image width
    uint32_t height;      // Image height
    uint32_t max_color;   // Max color intensity
    uint8_t *data;        // Pixel data (RGB format)
} PPMImage;

// Function to parse the PPM file
PPMImage *pixel_parse(const char *file_path) {
    FILE *file = fopen(file_path, "rb");
    if (file == NULL) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    PPMImage *image = malloc(sizeof(PPMImage));
    if (!image) {
        perror("Error allocating memory for PPMImage");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    // Read the PPM header
    if (fread(image->magic_number, sizeof(char), 2, file) != 2) {
        fprintf(stderr, "Error reading magic number\n");
        fclose(file);
        free(image); 
        exit(EXIT_FAILURE);
    }

    // Check if it's a valid P6 PPM file
    if (image->magic_number[0] != 'P' || image->magic_number[1] != '6') {
        fprintf(stderr, "Invalid PPM file format (not P6)\n");
        fclose(file);
        free(image);
        exit(EXIT_FAILURE);
    }

    // Read the image dimensions and max color intensity
    if (fscanf(file, "%u %u %u", &image->width, &image->height, &image->max_color) != 3) {
        fprintf(stderr, "Error reading image metadata\n");
        fclose(file);
        free(image);
        exit(EXIT_FAILURE);
    }

    // Skip the single whitespace character after the header
    fgetc(file);

    // Allocate memory for the pixel data
    size_t pixel_count = image->width * image->height;
    image->data = malloc(3 * pixel_count * sizeof(uint8_t)); // 3 bytes per pixel (RGB)
    if (!image->data) {
        perror("Error allocating memory for pixel data");
        fclose(file);
        free(image);
        exit(EXIT_FAILURE);
    }

    // Read the pixel data
    if (fread(image->data, sizeof(uint8_t), 3 * pixel_count, file) != 3 * pixel_count) {
        fprintf(stderr, "Error reading pixel data\n");
        fclose(file);
        free(image->data);
        free(image);
        exit(EXIT_FAILURE);
    }

    fclose(file);
    return image;
}

// Function to free the PPMImage memory
void free_ppm(PPMImage *image) {
    if (image) {
        free(image->data);
        free(image);
    }
}
