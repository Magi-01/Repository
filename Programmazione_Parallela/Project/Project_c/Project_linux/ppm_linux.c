/*
    Name: Fadhla Mohamed
    Sirname: Mutua
    Matricola: SM3201434
*/

#include "ppm_linux.h"
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <omp.h>

PPMImage *parse_ppm(const char *filename) {
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

    image->spheres.n = image->OBJ_N;
    image->spheres.cx = malloc(image->spheres.n * sizeof(float));
    image->spheres.cy = malloc(image->spheres.n * sizeof(float));
    image->spheres.cz = malloc(image->spheres.n * sizeof(float));
    image->spheres.r  = malloc(image->spheres.n * sizeof(float));
    image->spheres.color = malloc(image->spheres.n * sizeof(uint8_t[3]));
    if (!image->spheres.cx || !image->spheres.cy || !image->spheres.cz ||
    !image->spheres.r || !image->spheres.color) {
        fprintf(stderr, "Failed to allocate memory for spheres.\n");
        fclose(file);
        free(image);
        exit(1);
    }

    for (int i = 0; i < image->OBJ_N; i++) {
        if (fscanf(file, " S %f %f %f %f %hhu %hhu %hhu",
                &image->spheres.cx[i],
                &image->spheres.cy[i],
                &image->spheres.cz[i],
                &image->spheres.r[i],
                &image->spheres.color[i][0],
                &image->spheres.color[i][1],
                &image->spheres.color[i][2]) != 7) {
            fprintf(stderr, "Error reading sphere %d\n", i);
            fclose(file);
            free(image->spheres.cx);
            free(image->spheres.cy);
            free(image->spheres.cz);
            free(image->spheres.r);
            free(image->spheres.color);
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
            image->spheres.cx[i],
            image->spheres.cy[i],
            image->spheres.cz[i],
            image->spheres.r[i],
            image->spheres.color[i][0],
            image->spheres.color[i][1],
            image->spheres.color[i][2]);
    }

}

void free_image(PPMImage *image) {
    if (!image) return;

    free(image->spheres.cx);    // cx is float*
    free(image->spheres.cy);    // cy is float*
    free(image->spheres.cz);    // cz is float*
    free(image->spheres.r);     // r is float*
    free(image->spheres.color); // color is uint8_t (*)[3]

    // Free the PPMImage struct
    free(image);
}
