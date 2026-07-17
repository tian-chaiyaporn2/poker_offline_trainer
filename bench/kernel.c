// River showdown pass: the dominant per-iteration cost of a multi-street solve.
// For every complete board, compute both players' opponent-reach-weighted
// showdown utilities. This is the ~64k-evaluation inner loop, fused into one
// compiled call instead of ~1176 separate NumPy matvecs.
#include <stddef.h>

// One full pass over all boards (one CFR iteration's worth of river showdowns).
// E_all: [n_boards*no*ni] win matrix per board (0/0.5/1, compat baked in)
// B    : [no*ni] compatibility (board-independent)
// ri   : [ni] IP reach ; ro : [no] OOP reach
// uo/ui: [no]/[ni] accumulators (overwritten)
void river_pass(int n_boards, int no, int ni,
                const float* E_all, const float* B,
                const double* ri, const double* ro,
                double pot, double eo, double ei,
                double* uo, double* ui) {
    for (int i = 0; i < no; i++) uo[i] = 0.0;
    for (int j = 0; j < ni; j++) ui[j] = 0.0;
    for (int b = 0; b < n_boards; b++) {
        const float* E = E_all + (size_t)b * no * ni;
        // OOP utilities: uo[i] += pot*sum_j E[i][j]*ri[j] - eo*sum_j B[i][j]*ri[j]
        for (int i = 0; i < no; i++) {
            const float* Ei = E + (size_t)i * ni;
            const float* Bi = B + (size_t)i * ni;
            double se = 0.0, sb = 0.0;
            for (int j = 0; j < ni; j++) { se += (double)Ei[j] * ri[j]; sb += (double)Bi[j] * ri[j]; }
            uo[i] += pot * se - eo * sb;
        }
        // IP utilities: ui[j] += pot*sum_i (B-E)[i][j]*ro[i] - ei*sum_i B[i][j]*ro[i]
        for (int j = 0; j < ni; j++) {
            double se = 0.0, sb = 0.0;
            for (int i = 0; i < no; i++) {
                float Bij = B[(size_t)i * ni + j];
                float Eij = E[(size_t)i * ni + j];
                se += (double)(Bij - Eij) * ro[i];
                sb += (double)Bij * ro[i];
            }
            ui[j] += pot * se - ei * sb;
        }
    }
}
