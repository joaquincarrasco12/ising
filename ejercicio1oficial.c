#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define L 16
#define N (L*L)
#define EQUIL_STEPS 50000

#define J 1.0
#define B 0.0

void inicializar_spins(int S[L][L]) {
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            S[i][j] = (rand() % 2) * 2 - 1;
        }
    }
}

int magnetizacion(int S[L][L]) {
    int M = 0;
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            M += S[i][j];
        }
    }
    return M;
}

double energia(int S[L][L]) {
    double E = 0.0;
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            int derecha = S[i][(j + 1) % L];
            int abajo   = S[(i + 1) % L][j];
            E += -J * S[i][j] * (derecha + abajo);
        }
    }

    return E;
}

void metropolis(int S[L][L], double T) {
    double beta = 1.0 / T;
    int i = rand() % L;
    int j = rand() % L;
    int spin = S[i][j];
    
    int up    = S[(i - 1 + L) % L][j];
    int down  = S[(i + 1) % L][j];
    int left  = S[i][(j - 1 + L) % L];
    int right = S[i][(j + 1) % L];
    
    double dE = 2.0 * J * spin * (up + down + left + right);
    
    if (dE <= 0.0) {
        S[i][j] *= -1;
    } else {
        double r = (double) rand() / RAND_MAX;
        if (r < exp(-beta * dE)) {
            S[i][j] *= -1;
        }
    }
}

int main() {
    srand(time(NULL));
    
    int S[L][L];
    double T = 2.0;  // Temperatura fija
    
    inicializar_spins(S);
    
    FILE *file = fopen("ejercicio1.txt", "w");
    fprintf(file, "# Paso\tMagnetizacion\tEnergia\n");
    
    
    for (int step = 0; step < EQUIL_STEPS; step++) {
        metropolis(S, T);
        
        // Muestrear cada 100 pasos
        if (step % 100 == 0) {
            int M = magnetizacion(S);
            double E = energia(S);
            fprintf(file, "%d\t%d\t%f\n", step, M, E);
            
        }
    }
    
    fclose(file);
    
    return 0;
}