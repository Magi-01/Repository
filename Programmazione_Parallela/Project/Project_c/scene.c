#include "scene.h"
#include "ppm.h"
#include "calculations.h"
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

void set_pixel_color(PPMImage *image, uint32_t x, uint32_t y, uint32_t *color) {
    // Calculate the pixel index in the data array
    uint32_t pixel_index = (y * image->viewport[0] + x) * 3;

    // Set the RGB values
    image->data[pixel_index + 0] = color[0]; // Red
    image->data[pixel_index + 1] = color[1]; // Green
    image->data[pixel_index + 2] = color[2]; // Blue
}

void render_scene(PPMImage *image) {
    float origin[3] = {0.0f, 0.0f, 0.0f};  // Camera position

    // Only allocate memory for image data once
    if (&image->data == NULL) {
        image->data = (uint32_t *)malloc(image->viewport[0] * 
                    image->viewport[1] * 3 * sizeof(uint32_t));
        if (&image->data == NULL) {
            fprintf(stderr, "Memory allocation failed for image data.\n");
            exit(1); // Handle error appropriately
        }
    }

    // Iterate over the full width and height of the image
    for (uint32_t y = 0; y < image->viewport[1]; y++) {
        for (uint32_t x = 0; x < image->viewport[0]; x++) {
            // Map screen coordinates to viewport coordinates
            float viewport_coords[3];
            viewport_mapping(x, y, viewport_coords, image);

            // Compute ray direction
            float direction[3] = {viewport_coords[0], viewport_coords[1], viewport_coords[2]};
            normalize_ray(direction);

            // Calculate intersection with spheres and store the color
            uint32_t color[3];
            if (intersections(image, origin, image->OBJ_N, direction, image->spheres, x, y, color)) {
                // If a sphere is hit, use the color from the intersection
                set_pixel_color(image, x, y, color);
            } else {
                // If no intersection, use the background color
                uint32_t back_ground[3] = {image->background[0],image->background[1], image->background[2]};
                set_pixel_color(image, x, y, back_ground);
            }
        }
    }
}
