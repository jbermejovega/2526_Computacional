import numpy as np
from mpi4py import MPI
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import os
from numba import njit

# =============================================================================
# CONFIGURACIÓN MPI
# =============================================================================
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# =============================================================================
# PARÁMETROS FÍSICOS
# =============================================================================
N_SISTEMAS = 160
G = 1.0
M_AGUJERO_NEGRO_INICIAL = 4.1e6
M_SISTEMA = 1.0
R_COLISION = 0.06
R_ABSORCION = 0.5
R_FRONTERA = 15.0
DIST_INTERACCION = 5.0
DT = 0.0005
SOFTENING = 0.2
PASOS_ESTABILIZACION = 1000
PASOS_MEDICION = 1000
COLA = 150

OUTPUT_DIR = "resultados_mpi"

# =============================================================================
# DISTRIBUCIÓN DE CARGA
# =============================================================================
n_local = N_SISTEMAS // size
start_idx = rank * n_local
end_idx = (rank + 1) * n_local if rank != size - 1 else N_SISTEMAS

# =============================================================================
# FUNCIONES MATEMÁTICAS (Secuencial local con Plummer estricto)
# =============================================================================

@njit(fastmath=True)
def compute_local_acc(pos, start, end, m_bh, m_sys, r_limit_sq, softening, G_const):
    n_total = pos.shape[0]
    n_local = end - start
    local_acc = np.zeros((n_local, 2))
    
    # Calculamos el cuadrado del softening una sola vez por eficiencia
    soft_sq = softening**2
    
    for local_i in range(n_local):
        global_i = start + local_i
        
        rx = pos[global_i, 0]
        ry = pos[global_i, 1]
        r_sq = rx**2 + ry**2
        
        # FÓRMULA DE PLUMMER PARA EL AGUJERO NEGRO: M / (r^2 + e^2)^(3/2)
        factor_bh = -G_const * m_bh / ((r_sq + soft_sq)**1.5)
        local_acc[local_i, 0] = rx * factor_bh
        local_acc[local_i, 1] = ry * factor_bh

        for j in range(n_total):
            if global_i == j:
                continue
                
            dx = pos[j, 0] - pos[global_i, 0]
            dy = pos[j, 1] - pos[global_i, 1]
            d_sq = dx**2 + dy**2
            
            if d_sq < r_limit_sq:
                # FÓRMULA DE PLUMMER ENTRE ESTRELLAS: M / (r^2 + e^2)^(3/2)
                f_mag = G_const * m_sys / ((d_sq + soft_sq)**1.5)
                local_acc[local_i, 0] += f_mag * dx
                local_acc[local_i, 1] += f_mag * dy
                
    return local_acc

@njit(fastmath=True)
def resolver_colisiones_secuencial(pos, vel, r_col):
    n = pos.shape[0]
    r_double_sq = (2 * r_col)**2
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = pos[j, 0] - pos[i, 0], pos[j, 1] - pos[i, 1]
            d_sq = dx**2 + dy**2
            if 0 < d_sq < r_double_sq:
                dist = np.sqrt(d_sq)
                nx, ny = dx / dist, dy / dist
                rvx, rvy = vel[j, 0] - vel[i, 0], vel[j, 1] - vel[i, 1]
                v_imp = rvx * nx + rvy * ny
                if v_imp < 0:
                    vel[i, 0] += v_imp * nx; vel[i, 1] += v_imp * ny
                    vel[j, 0] -= v_imp * nx; vel[j, 1] -= v_imp * ny

# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================

def main():
    if rank == 0:
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        print(f"Iniciando simulación con {size} procesos MPI...")
        inicio = time.time()

        pos = np.random.uniform(-R_FRONTERA * 0.7, R_FRONTERA * 0.7, (N_SISTEMAS, 2))
        vel = np.zeros_like(pos)
        for i in range(N_SISTEMAS):
            r = np.sqrt(pos[i,0]**2 + pos[i,1]**2) or 1e-6
            v_mag = np.sqrt(G * M_AGUJERO_NEGRO_INICIAL / r) * np.random.uniform(0.7, 0.9)
            vel[i] = np.array([-pos[i, 1], pos[i, 0]]) / r * v_mag + np.random.normal(0, 0.1, 2)
            
        historial_inercia = []
        fotogramas = []
        fig_anim, ax_anim = plt.subplots(figsize=(7, 7))
    else:
        pos = None
        vel = None

    total_steps = PASOS_ESTABILIZACION + PASOS_MEDICION
    
    # Variable dinámica de masa
    m_bh_actual = M_AGUJERO_NEGRO_INICIAL

    # Sincronización inicial
    pos = comm.bcast(pos, root=0)
    m_bh_actual = comm.bcast(m_bh_actual, root=0)
    local_acc = compute_local_acc(pos, start_idx, end_idx, m_bh_actual, M_SISTEMA, DIST_INTERACCION**2, SOFTENING, G)
    gathered_acc = comm.gather(local_acc, root=0)
    
    if rank == 0:
        acc = np.vstack(gathered_acc)

    for step in range(total_steps):
        if rank == 0:
            v_half = vel + acc * DT / 2.0
            pos += v_half * DT

            for i in range(N_SISTEMAS):
                r_mag = np.sqrt(pos[i,0]**2 + pos[i,1]**2)
                
                # Absorción o fuga
                if r_mag < R_ABSORCION or r_mag > R_FRONTERA:
                    if r_mag < R_ABSORCION:
                        m_bh_actual += M_SISTEMA  # El agujero negro gana masa
                        
                    theta = np.random.uniform(0, 2 * np.pi)
                    pos[i] = R_FRONTERA * np.array([np.cos(theta), np.sin(theta)])
                    v_reg = np.sqrt(G * m_bh_actual / R_FRONTERA) * np.random.uniform(0.8, 0.95)
                    vel[i] = np.array([-pos[i, 1], pos[i, 0]]) / R_FRONTERA * v_reg
                    v_half[i] = vel[i]

            resolver_colisiones_secuencial(pos, v_half, R_COLISION)

        # Sincronización en cada paso
        pos = comm.bcast(pos, root=0)
        m_bh_actual = comm.bcast(m_bh_actual, root=0) 
        
        local_acc = compute_local_acc(pos, start_idx, end_idx, m_bh_actual, M_SISTEMA, DIST_INTERACCION**2, SOFTENING, G)
        gathered_acc = comm.gather(local_acc, root=0)

        if rank == 0:
            acc_new = np.vstack(gathered_acc)
            vel = v_half + acc_new * DT / 2.0
            acc = acc_new

            if step >= PASOS_ESTABILIZACION:
                I = np.sum(M_SISTEMA * np.sum(pos**2, axis=1))
                historial_inercia.append(I)

            if step % 80 == 0:
                p_pos = ax_anim.scatter(pos[:, 0], pos[:, 1], s=8, color='crimson', animated=True)
                p_bh = ax_anim.scatter(0, 0, s=100, color='black', animated=True)
                fotogramas.append([p_pos, p_bh])
                
            if step % 200 == 0:
                print(f"Progreso: {step}/{total_steps} pasos... Masa Agujero Negro: {m_bh_actual:.1f}")

    if rank == 0:
        print("Generando animación, esto puede tardar unos segundos...")
        ax_anim.set_xlim(-R_FRONTERA, R_FRONTERA)
        ax_anim.set_ylim(-R_FRONTERA, R_FRONTERA)
        ani = animation.ArtistAnimation(fig_anim, fotogramas, interval=40, blit=True)
        ani.save(os.path.join(OUTPUT_DIR, "simulacion_mpi.gif"), writer="pillow")
        plt.close(fig_anim)

        duracion = time.time() - inicio
        print("="*40)
        print("EJECUCIÓN MPI COMPLETADA")
        print(f"Nodos utilizados: {size}")
        print(f"Masa final del Agujero Negro: {m_bh_actual:.1f}")
        print(f"Tiempo total: {duracion:.2f} segundos")
        print("="*40)

if __name__ == "__main__":
    main()