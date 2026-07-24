/*
    Name: Fadhla Mohamed
    Sirname: Mutua
    Matricola: SM3201434
*/

#include "scene_linux.h"
#include "ppm_linux.h"
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <float.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <immintrin.h>
#include <assert.h>


// Function to compute the dot product of two vectors
float dot(const float *v1, const float *v2) {
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

/*
    Normalize a scalar 3D vector in-place.
    - ray: 3-element array (x,y,z)
    The length is sqrt(x^2 + y^2 + z^2).
    If length > 0, divides each component by the length to produce a unit vector.
*/
void normalize_ray_scalar(float *ray) {
    float len = sqrtf(ray[0]*ray[0] + ray[1]*ray[1] + ray[2]*ray[2]);
    if (len > 0.0f) {
        ray[0] /= len;
        ray[1] /= len;
        ray[2] /= len;
    }
}

/*
    Normalize 8 rays simultaneously using AVX (__m256 vectors).
    - dx, dy, dz: pointers to 8 floats each representing the components of 8 rays
    Computes len = sqrt(dx^2 + dy^2 + dz^2) element-wise and divides each component
    to produce 8 unit vectors in parallel.
*/
static inline void normalize_ray(
    __m256 *dx, __m256 *dy, __m256 *dz
) {
    __m256 len = _mm256_sqrt_ps(_mm256_add_ps(_mm256_add_ps(
                    _mm256_mul_ps(*dx,*dx),
                    _mm256_mul_ps(*dy,*dy)),
                    _mm256_mul_ps(*dz,*dz)));

    *dx = _mm256_div_ps(*dx, len);
    *dy = _mm256_div_ps(*dy, len);
    *dz = _mm256_div_ps(*dz, len);
}

/*
    Compute ray-sphere intersections for 8 rays at once (SIMD).
    Inputs:
        ox, oy, oz: ray origin (broadcasted)
        dx, dy, dz: ray directions (8 rays)
        cx, cy, cz: sphere center
        r: sphere radius
    Outputs:
        *t: closest intersection distances for each ray (8 floats)
    Returns:
        __m256 mask: element-wise mask (nonzero where intersection exists)
    
    Algorithm (for each ray):
    1. Compute vector from origin to sphere center: oc = C_s - O
    2. Compute coefficients:
        a = V·V (length squared of direction vector)
        b = -2 * (V · C_s)
        c = (C_s · C_s) - r^2
    3. Compute discriminant: disc = b^2 - 4*a*c
    4. If disc > 0, compute sqrt(disc) and t0, t1 = (-b ± sqrt(disc)) / (2a)
    5. Keep only positive t values and pick the minimum
*/
static inline __m256 ray_sphere_intersection(
    __m256 ox, __m256 oy, __m256 oz,
    __m256 dx, __m256 dy, __m256 dz,
    __m256 cx, __m256 cy, __m256 cz,
    __m256 r,
    __m256 *t
) {
    // Vector from ray origin to sphere center
    __m256 ocx = _mm256_sub_ps(cx, ox);
    __m256 ocy = _mm256_sub_ps(cy, oy);
    __m256 ocz = _mm256_sub_ps(cz, oz);

    // b = -2 * (V · C_s)
    __m256 b =
        _mm256_mul_ps(_mm256_set1_ps(-2.0f),
        _mm256_fmadd_ps(ocx, dx,
        _mm256_fmadd_ps(ocy, dy,
                        _mm256_mul_ps(ocz, dz))));

    // c = (C_s · C_s) - r^2
    __m256 c =
        _mm256_fmadd_ps(ocx, ocx,
        _mm256_fmadd_ps(ocy, ocy,
        _mm256_fmadd_ps(ocz, ocz,
                        _mm256_mul_ps(r, r))));

    // discriminant
    __m256 disc =
        _mm256_sub_ps(_mm256_mul_ps(b, b),
                       _mm256_mul_ps(_mm256_set1_ps(4.0f), c));

    // mask for rays with real intersections
    __m256 mask = _mm256_cmp_ps(disc, _mm256_setzero_ps(), _CMP_GT_OQ);

    // sqrt of discriminant
    __m256 zero = _mm256_setzero_ps();
    __m256 disc_pos = _mm256_max_ps(disc, zero);
    __m256 s = _mm256_sqrt_ps(disc_pos);

    // t0, t1 = (-b ± sqrt(d)) / 2
    __m256 two = _mm256_set1_ps(2.0f);
    __m256 t0 = _mm256_div_ps(_mm256_sub_ps(_mm256_setzero_ps(), _mm256_add_ps(b, s)), two);
    __m256 t1 = _mm256_div_ps(_mm256_sub_ps(_mm256_setzero_ps(), _mm256_sub_ps(b, s)), two);


    *t = _mm256_min_ps(t0, t1);
    return mask;
}

/*
    Scalar ray-sphere intersection for a single ray and single sphere.
    Inputs:
        origin: ray origin
        direction: ray direction (unit vector)
        spheres: scene spheres
        i: index of the sphere
    Outputs:
        *t: closest intersection distance
    Returns:
        1 if intersection exists, 0 otherwise
*/
int ray_sphere_intersection_scalar(const float *origin,
                                   const float *direction,
                                   const Sphere *spheres,
                                   int i,
                                   float *t)
{
    // Vector from origin to sphere center
    float oc[3] = {
        spheres->cx[i] - origin[0],
        spheres->cy[i] - origin[1],
        spheres->cz[i] - origin[2]
    };

    float a = 1.0f; // direction is normalized
    float b = -2.0f * (oc[0]*direction[0] + oc[1]*direction[1] + oc[2]*direction[2]);
    float c = oc[0]*oc[0] + oc[1]*oc[1] + oc[2]*oc[2] - spheres->r[i]*spheres->r[i];

    float disc = b*b - 4*a*c;

    if (disc < 0.0f) return 0; // no intersection

    float sqrt_d = sqrtf(disc);
    float t0 = (-b - sqrt_d) / (2*a);
    float t1 = (-b + sqrt_d) / (2*a);

    // Pick the closest positive t
    float tmin = FLT_MAX;
    if (t0 > 0.0f && t0 < tmin) tmin = t0;
    if (t1 > 0.0f && t1 < tmin) tmin = t1;

    if (tmin == FLT_MAX) return 0; // no positive intersection

    *t = tmin;
    return 1;
}

/*
    Scalar per-ray intersection with all spheres in scene.
    Inputs:
        image: scene
        origin: ray origin
        num_spheres: number of spheres
        direction: ray direction
        spheres: sphere array
        x, y: pixel coordinates (unused, but in signature)
    Outputs:
        color[3]: RGB value at intersection (or background)
    Returns:
        1 if a sphere was hit, 0 otherwise
*/
int intersections(PPMImage *image,
                  float *origin,
                  int num_spheres,
                  float *direction,
                  Sphere *spheres,
                  int x, int y,
                  uint8_t *color) {
    float closest_t = FLT_MAX;
    int hit_sphere = -1;

    for (int i = 0; i < num_spheres; i++) {
        float t;
        if (ray_sphere_intersection_scalar(origin, direction, spheres, i, &t)) {
            if (t < closest_t) {
                closest_t = t;
                hit_sphere = i;
            }
        }
    }

    if (hit_sphere != -1) {
        // Sphere hit
        color[0] = spheres->color[hit_sphere][0];
        color[1] = spheres->color[hit_sphere][1];
        color[2] = spheres->color[hit_sphere][2];
        return 1;
    }

    // No intersection then background
    color[0] = image->background[0];
    color[1] = image->background[1];
    color[2] = image->background[2];
    return 0;
}

/*
    SIMD intersections with all spheres
    Inputs:
        image: scene
        origin: camera origin
        vx, vy, vz: ray directions (8 rays)
        spheres: all scene spheres
        num_spheres: number of spheres
    Outputs:
        color_r, color_g, color_b: RGB for each of the 8 rays
*/
void intersections_simd(
    PPMImage *image,
    float *origin,
    __m256 vx, __m256 vy, __m256 vz,
    Sphere *spheres,
    int num_spheres,
    __m256i *color_r,
    __m256i *color_g,
    __m256i *color_b
) {
    const __m256 zero = _mm256_setzero_ps();
    const __m256 inf  = _mm256_set1_ps(FLT_MAX);
    const __m256 four = _mm256_set1_ps(4.0f);

    __m256 ox = _mm256_set1_ps(origin[0]);
    __m256 oy = _mm256_set1_ps(origin[1]);
    __m256 oz = _mm256_set1_ps(origin[2]);

    __m256 closest_t = inf;
    __m256i hit_idx  = _mm256_set1_epi32(-1);

    // a = V·V (per ray!)
    __m256 a = _mm256_add_ps(
                   _mm256_add_ps(
                       _mm256_mul_ps(vx, vx),
                       _mm256_mul_ps(vy, vy)),
                   _mm256_mul_ps(vz, vz));

    __m256 two_a = _mm256_add_ps(a, a);

    for (int s = 0; s < num_spheres; ++s) {

        __m256 cx = _mm256_set1_ps(spheres->cx[s]);
        __m256 cy = _mm256_set1_ps(spheres->cy[s]);
        __m256 cz = _mm256_set1_ps(spheres->cz[s]);
        __m256 r  = _mm256_set1_ps(spheres->r[s]);

        // C_s = C - O
        __m256 ocx = _mm256_sub_ps(cx, ox);
        __m256 ocy = _mm256_sub_ps(cy, oy);
        __m256 ocz = _mm256_sub_ps(cz, oz);

        // b = -2 * (V · C_s)
        __m256 dotVC = _mm256_add_ps(
                           _mm256_add_ps(
                               _mm256_mul_ps(vx, ocx),
                               _mm256_mul_ps(vy, ocy)),
                           _mm256_mul_ps(vz, ocz));

        __m256 b = _mm256_mul_ps(_mm256_set1_ps(-2.0f), dotVC);

        // c = (C_s · C_s) - r^2
        __m256 c = _mm256_sub_ps(
                       _mm256_add_ps(
                           _mm256_add_ps(
                               _mm256_mul_ps(ocx, ocx),
                               _mm256_mul_ps(ocy, ocy)),
                           _mm256_mul_ps(ocz, ocz)),
                       _mm256_mul_ps(r, r));

        // discriminant
        __m256 disc = _mm256_sub_ps(
                          _mm256_mul_ps(b, b),
                          _mm256_mul_ps(four, _mm256_mul_ps(a, c)));

        __m256 disc_mask = _mm256_cmp_ps(disc, zero, _CMP_GT_OQ);
        if (_mm256_testz_ps(disc_mask, disc_mask))
            continue;

        __m256 sqrt_d = _mm256_sqrt_ps(_mm256_max_ps(disc, zero));

        // t = (-b ± sqrt(d)) / (2a)
        __m256 t0 = _mm256_div_ps(
                        _mm256_sub_ps(_mm256_sub_ps(zero, b), sqrt_d),
                        two_a);

        __m256 t1 = _mm256_div_ps(
                        _mm256_add_ps(_mm256_sub_ps(zero, b), sqrt_d),
                        two_a);

        // keep positive t only
        __m256 t0p = _mm256_blendv_ps(inf, t0,
                        _mm256_cmp_ps(t0, zero, _CMP_GT_OQ));
        __m256 t1p = _mm256_blendv_ps(inf, t1,
                        _mm256_cmp_ps(t1, zero, _CMP_GT_OQ));

        __m256 t_min = _mm256_min_ps(t0p, t1p);

        __m256 valid = _mm256_and_ps(
            disc_mask,
            _mm256_cmp_ps(t_min, inf, _CMP_LT_OQ));

        __m256 closer = _mm256_cmp_ps(t_min, closest_t, _CMP_LT_OQ);
        __m256 update = _mm256_and_ps(valid, closer);

        closest_t = _mm256_blendv_ps(closest_t, t_min, update);

        __m256i mask_i = _mm256_castps_si256(update);
        __m256i s_i    = _mm256_set1_epi32(s);

        hit_idx = _mm256_blendv_epi8(hit_idx, s_i, mask_i);
    }

    // Assign RGB colors based on closest hits
    int idx[8];
    _mm256_storeu_si256((__m256i*)idx, hit_idx);
    uint32_t r[8], g[8], bcol[8];
    for (int i = 0; i < 8; i++) {
        if (idx[i] >= 0 && idx[i] < num_spheres) {
            r[i] = spheres->color[idx[i]][0];
            g[i] = spheres->color[idx[i]][1];
            bcol[i] = spheres->color[idx[i]][2];
        } else {
            r[i] = image->background[0];
            g[i] = image->background[1];
            bcol[i] = image->background[2];
        }
    }

    *color_r = _mm256_loadu_si256((__m256i*)r);
    *color_g = _mm256_loadu_si256((__m256i*)g);
    *color_b = _mm256_loadu_si256((__m256i*)bcol);
}


/*
    Map scalar pixel coordinates (x,y) to viewport coordinates.
    - width, height: image resolution
    - aspect_x, aspect_y: viewport dimensions in world space
    Returns viewport_coords[3] = (x, y, z=1)
*/
void viewport_mapping_scalar(int x, int y, float width, float height,
                             float aspect_x, float aspect_y,
                             float *viewport_coords) {
    viewport_coords[0] = x * aspect_x / (width - 1.0f) - aspect_x/2.0f;
    viewport_coords[1] = (height - y - 1.0f) * aspect_y / (height - 1.0f) - aspect_y/2.0f;
    viewport_coords[2] = 1.0f;
}


/*
    Map 8 pixels simultaneously to viewport coordinates (SIMD)
    Inputs:
        x, y: starting pixel coordinates
        w, h: image dimensions
        ax, ay: viewport scaling
    Outputs:
        vx, vy: __m256 vectors of x and y coordinates
*/
static inline void viewport_mapping(
    int x, int y,
    float w, float h,
    float ax, float ay,
    __m256 *vx, __m256 *vy
) {
    // Pixel centers
    __m256 xs = _mm256_set_ps(x+7,x+6,x+5,x+4,
                              x+3,x+2,x+1,x);

    *vx = _mm256_sub_ps(_mm256_mul_ps(_mm256_set1_ps(ax/(w-1.0f)), xs),
                        _mm256_set1_ps(ax/2.0f));

    float fy_scalar = (h - y - 1.0f) * ay / (h - 1.0f) - ay/2.0f;
    *vy = _mm256_set1_ps(fy_scalar);
}


/*
    Render the scene using OpenMP and memory-mapped file.
    - Each thread processes rows of pixels.
    - 8 pixels at a time are handled with SIMD.
    - Remaining pixels in a row are handled with scalar fallback.
    - Writes PPM image directly to disk.
*/
void render_scene(char *output_file, PPMImage *image, float width, float height) {
    // Set the number of threads
    omp_set_num_threads(6);
    int header_size = snprintf(NULL, 0, "P6\n%u %u\n255\n",
                           (int)width, (int)height);

    printf("output file: %s\n", output_file);

    /* Open or create the file */
    int fd = open(output_file, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd == -1) {
        perror("open");
        exit(1);
    }

    /* Compute final file size */
    size_t file_size = header_size + 3 * (size_t)width * (size_t)height;

    /* Resize file to required size */
    if (ftruncate(fd, file_size) == -1) {
        perror("ftruncate");
        close(fd);
        exit(1);
    }

    /* Map file into memory */
    uint8_t *mapped_memory = mmap(
        NULL,                // Let kernel choose address
        file_size,           // Mapping size
        PROT_READ | PROT_WRITE,
        MAP_SHARED,           // Changes propagate to file
        fd,
        0
    );

    if (mapped_memory == MAP_FAILED) {
        perror("mmap");
        close(fd);
        exit(1);
    }

    /* Write PPM header */
    snprintf((char *)mapped_memory,
            header_size + 1,
            "P6\n%u %u\n255\n",
            (int)width, (int)height);

    /* Pixel data starts after header */
    uint8_t *pixel_data = mapped_memory + header_size;

    assert(image->spheres.cx != NULL);
    assert(image->spheres.cy != NULL);
    assert(image->spheres.cz != NULL);
    assert(image->spheres.r != NULL);
    assert(image->spheres.color != NULL);

    // Prepare the scene for rendering
    float origin[3] = {0.0f, 0.0f, 0.0f};  // Camera position
    float viewport_width = width; // e.g. 1920
    float viewport_height = height; // e.g. 1080
    int num_spheres = image->OBJ_N; //number of objects
    const float aspect_x = image->viewport[0]; //width aspect ratio
    const float aspect_y = image->viewport[1]; //height aspect ratio

    /*
    Parallelize the rendering of pixels using OpenMP.

    The "for" loop is divided into chunks with static scheduling

    "collapse(2)" combines the two nested loops (over x and y) into a single loop.

    "default(none)" ensures that all variables used inside the parallel block must be explicitly specified as shared.

    "shared(image, pixel_data, origin, viewport_width, viewport_height, num_spheres, aspect_x, aspect_y)"
    indicates that these variables are shared between all threads, meaning they are accessed by all threads and do not have separate copies for each thread.
    */

    #pragma omp parallel for schedule(static) default(none) \
        shared(image, pixel_data, origin, viewport_width, viewport_height, num_spheres, aspect_x, aspect_y)
    for (int y = 0; y < (int)viewport_height; y++) {

        // Process 8 pixels at a time from AVX2 being able to compute 32bit chunks
        int x;
        for (x = 0; x <= (int)viewport_width - 8; x += 8) {

            // SIMD coordinates
            __m256 vx, vy, vz;
            viewport_mapping(x, y, viewport_width, viewport_height,
                            aspect_x, aspect_y, &vx, &vy);
            vz = _mm256_set1_ps(image->viewport[2]);  // z = 1.0

            // Normalize 8 rays
            normalize_ray(&vx, &vy, &vz);

            // Allocate temporary colors (8 pixels)
            __m256i color_r, color_g, color_b;

            // Call SIMD intersections
            intersections_simd(image, origin, vx, vy, vz,
                            &image->spheres, num_spheres,
                            &color_r, &color_g, &color_b);

            // Stores the 
            uint32_t tmp_r[8], tmp_g[8], tmp_b[8];
            _mm256_storeu_si256((__m256i*)tmp_r, color_r);
            _mm256_storeu_si256((__m256i*)tmp_g, color_g);
            _mm256_storeu_si256((__m256i*)tmp_b, color_b);

            // Store 8 pixels into memory
            for (int i = 0; i < 8; i++) {
                size_t idx = ((size_t)y * (size_t)viewport_width + (size_t)(x + i)) * 3;
                pixel_data[idx + 0] = (uint8_t)tmp_r[i];
                pixel_data[idx + 1] = (uint8_t)tmp_g[i];
                pixel_data[idx + 2] = (uint8_t)tmp_b[i];
            }
        }

        for (; x < (int)viewport_width; x++) {
            // Scalar viewport coordinates
            float viewport_coords[3];
            viewport_mapping_scalar(x, y, viewport_width, viewport_height,
                                    aspect_x, aspect_y, viewport_coords);

            // Scalar ray direction
            float direction[3] = {
                viewport_coords[0],
                viewport_coords[1],
                viewport_coords[2]
            };
            normalize_ray_scalar(direction); // scalar normalization

            // Compute intersections
            uint8_t color[3];
            intersections(image, origin, num_spheres, direction, &image->spheres,
                        x, y, color);

            // Store in pixel buffer
            size_t idx = ((size_t)y * (size_t)viewport_width + (size_t)x) * 3;
            pixel_data[idx + 0] = color[0];
            pixel_data[idx + 1] = color[1];
            pixel_data[idx + 2] = color[2];
        }
    }

    // Close the mapping and file handles
    munmap(mapped_memory, file_size);
    close(fd);
}

