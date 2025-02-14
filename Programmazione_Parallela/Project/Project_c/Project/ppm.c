/*
    Name: Fadhla Mohamed
    Sirname: Mutua
    Matricola: SM3201434
*/

#include "ppm.h"
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include <omp.h>

PPMImage *parse_ppm(const char *filename, const float width, const float height) {
    // Open the file (.txt) in read mode
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Failed to open file: %s. Check filename or location\n", filename);
        exit(1);
    }

    // Allocate memory for image
    PPMImage *image = (PPMImage *)malloc(sizeof(PPMImage));
    if (!image) {
    fprintf(stderr, "Failed to allocate memory for image.\n");
    fclose(file);
    free(image);
    exit(1);
    }

    // Read Format
    if (fscanf(file, "%s", image->magic_number) != 1) {
        fprintf(stderr, "Error reading magic number.\n");
        fclose(file);
        free(image);
        exit(1);
    }

    // Read viewport
    if (fscanf(file, " VP %f %f %f", &image->viewport[0], &image->viewport[1], &image->viewport[2]) != 3) {
        fprintf(stderr, "Error reading viewport.\n");
        fclose(file);
        free(image);
        exit(1);
    }

    // Read background
    if (fscanf(file, " BG %hhu %hhu %hhu", &image->background[0], &image->background[1], &image->background[2]) != 3) {
        fprintf(stderr, "Error reading background.\n");
        fclose(file);
        free(image);
        exit(1);
    }

    // Read number of objects
    if (fscanf(file, " OBJ_N %d", &image->OBJ_N) != 1) {
        fprintf(stderr, "Error reading number of objects.\n");
        fclose(file);
        free(image);
        exit(1);
    }

    // Allocate memory for spheres
    image->spheres = (Sphere *)malloc(image->OBJ_N * sizeof(Sphere));
    if (!image->spheres) {
    fprintf(stderr, "Failed to allocate memory for spheres.\n");
    fclose(file);
    free(image);
    exit(1);
    }

    // Read spheres data
    int i;
    for (i = 0; i < image->OBJ_N; i++) {
        if (fscanf(file, " S %f %f %f %f %hhu %hhu %hhu",
                &image->spheres[i].center[0],
                &image->spheres[i].center[1],
                &image->spheres[i].center[2],
                &image->spheres[i].radius,
                &image->spheres[i].color[0],
                &image->spheres[i].color[1],
                &image->spheres[i].color[2]) != 7) {
            fprintf(stderr, "Error reading sphere %d.\n", i + 1);
            free(image->spheres);
            fclose(file);
            free(image);
            exit(1);
        }
    }

    fclose(file);
    return image;
}

void print_ppm_data(PPMImage *image) {
    // Print magic number
    printf("Magic Number: %c%c\n", image->magic_number[0], image->magic_number[1]);
    
    // Print viewport dimensions
    printf("Viewport: %.2f %.2f %.2f\n", image->viewport[0], image->viewport[1], image->viewport[2]);
    
    // Print background color
    printf("Background color: %u %u %u\n", image->background[0], image->background[1], image->background[2]);
    
    // Print number of objects (spheres)
    printf("Number of spheres: %d\n", image->OBJ_N);
    
    // Print sphere data
    for (int i = 0; i < image->OBJ_N; ++i) {
        printf("Sphere %d: Position (%.2f, %.2f, %.2f), Radius %.2f, RGB (%hhu, %hhu, %hhu)\n", 
               i + 1, 
               image->spheres[i].center[0], image->spheres[i].center[1], image->spheres[i].center[2], 
               image->spheres[i].radius,
               image->spheres[i].color[0], image->spheres[i].color[2],
               image->spheres[i].color[2]);
    }

}