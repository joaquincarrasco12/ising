#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define L 16
#define N (L*L)
#define EQUIL_STEPS 50000
#define MEASURE_STEPS 100000
#define SAMPLE_INTERVAL 10

#define J 1.0
#define B 0.0

const double temperaturas[] = {1.5, 1.8, 2.0, 2.1, 2.2, 2.25, 2.27, 2.3, 2.4, 2.6, 3.0, 4.0};
const int n_temps = sizeof(temperaturas) / sizeof(temperaturas[0]);

void inicializar_spins(int S[L][L], unsigned int *seed) {
    for (int i = 0; i < L; i++)
        for (int j = 0; j < L; j++)
            S[i][j] = (rand_r(seed) % 2) * 2 - 1;
}

int magnetizacion(int S[L][L]) {
    int M = 0;
    for (int i = 0; i < L; i++)
        for (int j = 0; j < L; j++)
            M += S[i][j];
    return M;
}

void metropolis(int S[L][L], double T, unsigned int *seed) {
    double beta = 1.0 / T;
    int i = rand_r(seed) % L;
    int j = rand_r(seed) % L;
    int spin = S[i][j];
    
    int up    = S[(i - 1 + L) % L][j];
    int down  = S[(i + 1) % L][j];
    int left  = S[i][(j - 1 + L) % L];
    int right = S[i][(j + 1) % L];
    
    double dE = 2.0 * J * spin * (up + down + left + right);
    
    if (dE <= 0.0) {
        S[i][j] *= -1;
    } else {
        double r = (double) rand_r(seed) / RAND_MAX;
        if (r < exp(-beta * dE))
            S[i][j] *= -1;
    }
}

void calcular_correlacion(int S[L][L], double *corr, int max_dist) {
    for (int d = 0; d <= max_dist; d++)
        corr[d] = 0.0;
    
    int ref_i = L/2, ref_j = L/2;
    int spin_ref = S[ref_i][ref_j];
    int *count = calloc(max_dist + 1, sizeof(int));
    
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            int dist = abs(i - ref_i) + abs(j - ref_j);
            if (dist <= max_dist && dist > 0) {
                corr[dist] += spin_ref * S[i][j];
                count[dist]++;
            }
        }
    }
    
    double prom_spin = (double)magnetizacion(S) / N;
    for (int d = 1; d <= max_dist; d++)
        if (count[d] > 0)
            corr[d] = corr[d] / count[d] - prom_spin * prom_spin;
    
    free(count);
}

double obtener_longitud_correlacion(double *corr, int max_dist) {
    int n_points = 0;
    double sum_R = 0.0, sum_logC = 0.0, sum_R2 = 0.0, sum_R_logC = 0.0;
    
    for (int R = 1; R <= max_dist; R++) {
        if (corr[R] > 1e-6) {
            double logC = log(corr[R]);
            sum_R += R;
            sum_logC += logC;
            sum_R2 += R * R;
            sum_R_logC += R * logC;
            n_points++;
        }
    }
    
    if (n_points < 2) return 0.0;
    
    double pendiente = (n_points * sum_R_logC - sum_R * sum_logC) / 
                       (n_points * sum_R2 - sum_R * sum_R);
    
    return -1.0 / pendiente;
}

int main() {
    int max_dist = L/2;
    
    // Archivo de salida (escritura con protección)
    FILE *file = fopen("longitud_correlacion.txt", "w");
    fprintf(file, "# Temperatura\tLongitud_Correlacion\n");
    
    // Paralelización por temperatura
    #pragma omp parallel for schedule(dynamic)
    for (int t_idx = 0; t_idx < n_temps; t_idx++) {
        double T = temperaturas[t_idx];
        
        // Semilla única por hilo y temperatura
        unsigned int seed = time(NULL) ^ omp_get_thread_num() ^ (t_idx * 12345);
        
        int S[L][L];
        
        printf("Hilo %d procesando T = %.3f...\n", omp_get_thread_num(), T);
        
        inicializar_spins(S, &seed);
        
        // Equilibración
        for (int step = 0; step < EQUIL_STEPS; step++)
            metropolis(S, T, &seed);
        
        double *corr_sum = calloc(max_dist + 1, sizeof(double));
        int n_medidas = 0;
        
        // Mediciones
        for (int step = 0; step < MEASURE_STEPS; step++) {
            metropolis(S, T, &seed);
            
            if (step % SAMPLE_INTERVAL == 0) {
                double *corr = calloc(max_dist + 1, sizeof(double));
                calcular_correlacion(S, corr, max_dist);
                for (int d = 1; d <= max_dist; d++)
                    corr_sum[d] += corr[d];
                n_medidas++;
                free(corr);
            }
        }
        
        double *corr_promedio = calloc(max_dist + 1, sizeof(double));
        for (int d = 1; d <= max_dist; d++)
            corr_promedio[d] = corr_sum[d] / n_medidas;
        
        double xi = obtener_longitud_correlacion(corr_promedio, max_dist);
        
        // Escritura al archivo (sección crítica para evitar conflictos)
        #pragma omp critical
        {
            fprintf(file, "%.3f\t%.6f\n", T, xi);
            printf("Hilo %d: T = %.3f, ξ = %.4f\n", omp_get_thread_num(), T, xi);
        }
        
        free(corr_sum);
        free(corr_promedio);
    }
    
    fclose(file);
    printf("\nSimulación completada. Resultados en 'longitud_correlacion.txt'\n");
    return 0;
}