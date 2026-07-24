/*
    Name: Fadhla Mohamed
    Sirname: Mutua
    Matricola: SM3201434
*/

#include "scene.h"
#include "ppm.h"
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <float.h>
#include <windows.h>

// Function to compute the dot product of two vectors
float dot(const float *v1, const float *v2) {
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

// Function to normalize a ray direction
void normalize_ray(float *ray) {
    float length = sqrt(dot(ray,ray));
    if (length > 0) {
        ray[0] /= length;
        ray[1] /= length;
        ray[2] /= length;
    }
}

// Function to check for ray-sphere intersection
int ray_sphere_intersection(const float *origin, const float *direction, const Sphere *sphere, float *t, int x, int y) {
    // Taking the vectors for sphere respect to the origin
    float oc[3] = {
        sphere->center[0] - origin[0],
        sphere->center[1] - origin[1],
        sphere->center[2] - origin[2]
    };
    // a = V * V (dot product of normalised vector)
    float a = dot(direction, direction);
    // b = -2* (oc * V) (dot product between normalised vector and spheres)
    float b = -2.0f * dot(oc, direction);
    // c = (oc * oc) - radius^2 (dot product between sphere displaced by r^2)
    float c = dot(oc, oc) - sphere->radius * sphere->radius;
    // Discriminant = b^2 - 4*a*c
    float discriminant = b * b - 4 * a * c;

    /*
    The two solutions for t in V^2t^2 -2*V*oc +(oc^2 - r^2) = 0
    Their possibilities are
    1.) discriminant < 0 -> no solutions 
    2.) discriminant = 0 -> one solutions
    3.) discriminant > 0 -> two solutions
    */
    if (discriminant < 0) {
        return 0;  // No intersection
    }
    
    // If discriminant is positive, there are two intersections
    float sqrt_d = sqrtf(fmaxf(discriminant, 0.0f)); // Ensure no negative sqrt
    float t0 = (-b - sqrt_d) / (2.0f * a);
    float t1 = (-b + sqrt_d) / (2.0f * a);

    *t = fminf(t0 * (t0 > 0) + t1 * (t1 > 0), FLT_MAX);
    return *t < FLT_MAX;
}

// Function to find intersections with spheres
int intersections(PPMImage *image, float *origin, int num_spheres, float *direction, Sphere *spheres, int x, int y, uint8_t *color) {
    /*
    Initialize closest_t to the maximum possible float value (FLT_MAX).
    This variable will store the smallest distance (t) at which the ray intersects a sphere.
    */
    float closest_t = FLT_MAX;

    // Initialize hit_sphere_index to -1 to indicate no sphere has been hit initially.
    int hit_sphere_index = -1;

    // Loop through each sphere in the scene (num_spheres)
    for (int i = 0; i < num_spheres; i++) {
        // Declare a variable t to store the intersection distance with the current sphere
        float t;

        // Check if the ray intersects with the current sphere
        if (ray_sphere_intersection(origin, direction, &spheres[i], &t, x, y)) {
            // If the intersection distance t is closer than the current closest_t
            if (t < closest_t) {
                // Update closest_t to the new, smaller intersection distance
                closest_t = t;
                // Update hit_sphere_index to the current sphere index, indicating this is the closest sphere hit
                hit_sphere_index = i;
            }
        }
    }

    if (hit_sphere_index != -1) {
        // A sphere was hit; return its color
        color[0] = spheres[hit_sphere_index].color[0];
        color[1] = spheres[hit_sphere_index].color[1];
        color[2] = spheres[hit_sphere_index].color[2];
        return 1;
    }

    // No intersection; return background color
    color[0] = image->background[0];
    color[1] = image->background[1];
    color[2] = image->background[2];
    return 0;
}


// Function to map pixel coordinates to viewport coordinates
void viewport_mapping(float x_pixel, float y_pixel, float *viewport_coords, PPMImage *image, float width, float height) {
    // Get aspect ratios
    float aspect_ratio_x = image->viewport[0];
    float aspect_ratio_y = image->viewport[1];

    /*
    Adjust x and y to match the aspect ratios (y is mirrored due to screen coordinate system)
    The x and y pixel coordinates are mapped from screen space to viewport space, and the aspect ratios are applied to maintain the correct proportions.
    */
    viewport_coords[0] = (2.0f * x_pixel / (width - 1.0f) - 1.0f) * aspect_ratio_x;
    viewport_coords[1] = (2.0f * (height - y_pixel) / (height - 1.0f) - 1.0f) * aspect_ratio_y; // Adjust y for aspect ratio and mirror vertically
    viewport_coords[2] = 1.0f;
}


// Function to render the scene using mmap (windows CreateFileMapping) for writing the PPM file
void render_scene(char *output_file, PPMImage *image, float width, float height) {
    // Set the number of threads
    omp_set_num_threads(6);
    int header_size = snprintf(NULL, 0, "P6\n%u %u\nBG 255\n", (int)width, (int)height);

    // Open the file using CreateFile to create or open the file for writing
    /*  
        output_file,  // Name of the output file

        GENERIC_READ | GENERIC_WRITE, // Desired access (read and write permissions)

        0,   // Sharing mode (0 means the file cannot be shared/accessed by other processes)

        NULL,    // Security attributes (NULL means default security settings)

        CREATE_ALWAYS,   // Creation disposition (always create a new file, overwriting if it exists)

        FILE_ATTRIBUTE_NORMAL,  // File attributes (normal file with no special attributes, e.g., it's not hidden)

        NULL    // Template file (NULL means no template is used)
    */
    printf("output file: %s", output_file);
    HANDLE hFile = CreateFile(output_file, GENERIC_READ | GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        printf("Failed to open output file\n");
        exit(1);
    }

    /*
    Calculate the size of the file
    The header consists of a fixed size (20 bytes) for the "P6\n%u %u\n255\n" part, plus the size of the dynamic header.
    The RGB pixel data will be 3 bytes per pixel (for each color channel).
    */
    size_t file_size = 3 * width * height + header_size; // 3 bytes per pixel for RGB + 20 bytes + dynamic header

    // Create a file mapping object
    /*  
        hFile,        // Handle to the file that will be mapped

        NULL,         // Security attributes (NULL means default security settings)

        PAGE_READWRITE, // Protection mode (allows read and write access)

        0,            // high-order part of the maximum size of the file mapping(0 for files smaller than 4GB)

        file_size,    // Low-order DWORD of maximum file size (i.e. actual size of the mapped region)

        NULL          // Name of the mapping object (NULL means it is unnamed)
    */
    HANDLE hMapping = CreateFileMapping(hFile, NULL, PAGE_READWRITE, 0, file_size, NULL);
    if (hMapping == NULL) {
        printf("Failed to create file mapping\n");
        CloseHandle(hFile);
        exit(1);
    }

    // Map the file into memory
    uint8_t *mapped_memory = (uint8_t *)MapViewOfFile(hMapping, FILE_MAP_WRITE, 0, 0, file_size);
    if (mapped_memory == NULL) {
        printf("Failed to map file into memory\n");
        CloseHandle(hMapping);
        CloseHandle(hFile);
        exit(1);
    }

    // Write the PPM header directly to the mapped memory
    snprintf((char *)mapped_memory, header_size, "P6\n%u %u\n255\n", (int)width, (int)height);

    // Pixel data starts after the header
    uint8_t *pixel_data = mapped_memory + header_size;

    // Prepare the scene for rendering
    float origin[3] = {0.0f, 0.0f, 0.0f};  // Camera position
    float viewport_width = width; // e.g. 1920
    float viewport_height = height; // e.g. 1080
    float num_spheres = image->OBJ_N; //number of objects

    /*
    Parallelize the rendering of pixels using OpenMP.

    The "for" loop is divided into chunks with dynamic scheduling, where     each thread will process a 2x2 block of pixels at a time.

    "collapse(2)" combines the two nested loops (over x and y) into a single loop.

    "default(none)" ensures that all variables used inside the parallel block must be explicitly specified as either shared or private.

    "shared(image, pixel_data, origin)" indicates that these variables are shared between all threads, meaning they are accessed by all threads and do not have separate copies for each thread.

    "firstprivate(viewport_width, viewport_height, num_spheres)" makes sure that each thread gets a private copy of these variables with the initial values from the master thread, preventing data race issues.
    */
    #pragma omp parallel for schedule(dynamic, 2) collapse(2) default(none) \
            shared(image, pixel_data, origin) firstprivate(viewport_width, viewport_height, num_spheres)

    for (int y = 0; y < (int)viewport_height; y++) {
        for (int x = 0; x < (int)viewport_width; x++) {
            uint8_t color[3];

            // Map screen coordinates to viewport coordinates
            float viewport_coords[3];
            viewport_mapping(x, y, viewport_coords, image, viewport_width, viewport_height);

            // Out-of-bounds check
            uint8_t out_of_bounds = (viewport_coords[0] < -1.0f ||
                                    viewport_coords[0] > 1.0f || 
                                    viewport_coords[1] < -1.0f || viewport_coords[1] > 1.0f);
            color[0] = 255 * out_of_bounds; // Red
            color[1] = 255 * out_of_bounds; // Green
            color[2] = 255 * out_of_bounds; // Blue

            // Only calculate intersections if inside the viewport
            if (!out_of_bounds) {
                float direction[3] = {viewport_coords[0], viewport_coords[1], viewport_coords[2]};
                normalize_ray(direction);

                intersections(image, origin, num_spheres, direction, image->spheres, x, y, color);
            }

            size_t pixel_index = ((size_t)y * (size_t)viewport_width + (size_t)x) * 3;
            pixel_data[pixel_index] = color[0];
            pixel_data[pixel_index + 1] = color[1];
            pixel_data[pixel_index + 2] = color[2];
        }
    }

    // Synchronize and unmap the memory
    if (!FlushViewOfFile(mapped_memory, file_size)) {
        printf("Failed to flush mapped memory\n");
        UnmapViewOfFile(mapped_memory);
        CloseHandle(hMapping);
        CloseHandle(hFile);
        exit(1);
    }
    if (!UnmapViewOfFile(mapped_memory)) {
        printf("Failed to unmap file\n");
        CloseHandle(hMapping);
        CloseHandle(hFile);
        exit(1);
    }

    // Close the mapping and file handles
    CloseHandle(hMapping);
    CloseHandle(hFile);
}

