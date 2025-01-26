#include "ppm.h"
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

PPMImage parse_ppm(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        exit(1);
    }

    PPMImage image;
    fscanf(file, "%s", image.magic_number); // "P6"
    fscanf(file, "VP %f %f %f", &image.viewport[0], &image.viewport[1], &image.viewport[2]);
    fscanf(file, "BG %d %d %d", &image.background[0], &image.background[1], &image.background[2]);
    fscanf(file, "OBJ_N %d", &image.OBJ_N);

    // Allocate memory for spheres
    image.spheres = (Sphere *)malloc(image.OBJ_N * sizeof(Sphere));

    // Read spheres data
    for (int i = 0; i < image.OBJ_N; i++) {
        fscanf(file, "S %f %f %f %f %d %d %d", 
            &image.spheres[i].center[0], 
            &image.spheres[i].center[1], 
            &image.spheres[i].center[2], 
            &image.spheres[i].radius,
            &image.spheres[i].color[0], 
            &image.spheres[i].color[1], 
            &image.spheres[i].color[2]);
    }

    fclose(file);
    return image;
}

void save_ppm_image(const char *filename, PPMImage *image) {
    // Calculate the total size of the file to write
    size_t header_size = snprintf(NULL, 0, "%c%c\n%u %u\n%u\n", 
                                   image->magic_number[0], image->magic_number[1], 
                                   image->viewport[0], image->viewport[1], 
                                   image->OBJ_N);
    size_t data_size = image->viewport[0] * image->viewport[1] * 3; // Pixel data size
    size_t total_size = header_size + data_size;

    // Open the file for writing
    HANDLE hFile = CreateFile(filename, GENERIC_WRITE | GENERIC_READ, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "Failed to create or open file: %s\n", filename);
        exit(1);
    }

    // Set the file size
    LARGE_INTEGER liSize;
    liSize.QuadPart = total_size;
    if (!SetFilePointerEx(hFile, liSize, NULL, FILE_BEGIN) || !SetEndOfFile(hFile)) {
        fprintf(stderr, "Failed to set file size: %s\n", filename);
        CloseHandle(hFile);
        exit(1);
    }

    // Create a file mapping object
    HANDLE hMapping = CreateFileMapping(hFile, NULL, PAGE_READWRITE, 0, 0, NULL);
    if (!hMapping) {
        fprintf(stderr, "Failed to create file mapping: %s\n", filename);
        CloseHandle(hFile);
        exit(1);
    }

    // Map the file to memory
    char *mapped_file = (char *)MapViewOfFile(hMapping, FILE_MAP_WRITE, 0, 0, 0);
    if (!mapped_file) {
        fprintf(stderr, "Failed to map file to memory: %s\n", filename);
        CloseHandle(hMapping);
        CloseHandle(hFile);
        exit(1);
    }

    // Write the header to the mapped memory
    snprintf(mapped_file, header_size + 1, "%c%c\n%u %u\n%u\n", 
             image->magic_number[0], image->magic_number[1], 
             image->viewport[0], image->viewport[1], 
             image->OBJ_N);

    // Copy the pixel data to the mapped memory (after the header)
    memcpy(mapped_file + header_size, image->data, data_size);

    // Unmap the file and close handles
    UnmapViewOfFile(mapped_file);
    CloseHandle(hMapping);
    CloseHandle(hFile);

    // Free dynamically allocated resources
    free(image->data);
    free(image->spheres);
}

