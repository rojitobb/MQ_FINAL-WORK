import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

class PhotoexcitedRPM:
    """
    Simulación de la fotoexcitación y formación de pares radicales singlete
    Usando solo NumPy y SciPy
    """

    def __init__(self):
        self.setup_system()

    def setup_system(self):
        """Configura el sistema molecular para fotoexcitación"""
        self.dim = 4
        # Estados: |0⟩=ground, |1⟩=excited, |2⟩=RP singlet, |3⟩=RP triplet

        # Vectores base
        self.ground = np.array([1, 0, 0, 0], dtype=complex)      # |G⟩
        self.excited = np.array([0, 1, 0, 0], dtype=complex)     # |E⟩
        self.rp_singlet = np.array([0, 0, 1, 0], dtype=complex)  # |RP_S⟩
        self.rp_triplet = np.array([0, 0, 0, 1], dtype=complex)  # |RP_T⟩

        self.H0 = self.create_hamiltonian()

    def create_hamiltonian(self):
        """Crea el Hamiltoniano del sistema"""
        # Energías de los estados (unidades arbitrarias)
        E_g = 0.0      # Energía del estado fundamental
        E_e = 2.0      # Energía del estado excitado
        E_rp = 1.5     # Energía del par radical

        H = np.diag([E_g, E_e, E_rp, E_rp])
        return H

    def photoexcitation_pulse(self, duration=1.0, intensity=1.0):
        """
        Simula un pulso de fotoexcitación que promueve |G⟩ → |E⟩
        """
        # Operador de transición |G⟩⟨E| + |E⟩⟨G|
        transition_op = np.outer(self.ground, self.excited.conj()) + np.outer(self.excited, self.ground.conj())

        # Hamiltoniano durante el pulso
        H_pulse = self.H0 + intensity * transition_op

        # Evolucionar el sistema durante el pulso
        U_pulse = expm(-1j * H_pulse * duration)

        return U_pulse

    def charge_separation(self, excited_state, efficiency=0.9):
        """
        Simula la separación de carga que forma el par radical
        """
        # Probabilidad de formar par radical singlete vs triplete
        p_singlet = efficiency * 0.95  # Alta eficiencia para singlete
        p_triplet = efficiency * 0.05  # Baja probabilidad de triplete inicial

        # Crear matriz densidad final
        rho_final = np.zeros((self.dim, self.dim), dtype=complex)

        # Añadir componentes
        rho_final += p_singlet * np.outer(self.rp_singlet, self.rp_singlet.conj())
        rho_final += p_triplet * np.outer(self.rp_triplet, self.rp_triplet.conj())
        rho_final += (1 - efficiency) * np.outer(self.excited, self.excited.conj())

        return rho_final

    def simulate_photoexcitation_process(self):
        """Simula el proceso completo de fotoexcitación y formación de par radical"""
        print("=== SIMULACIÓN DE FOTOEXCITACIÓN DE PARES RADICALES ===\n")

        # 1. Estado inicial en el ground state
        initial_state = self.ground.copy()
        print(f"1. Estado inicial: |G⟩")
        print(f"   Población en |G⟩: {np.abs(initial_state[0])**2:.3f}")

        # 2. Aplicar pulso de fotoexcitación
        U_pulse = self.photoexcitation_pulse(duration=0.5, intensity=2.0)
        state_after_pulse = U_pulse @ initial_state

        print(f"\n2. Después del pulso de luz:")
        print(f"   Población en |G⟩: {np.abs(state_after_pulse[0])**2:.3f}")
        print(f"   Población en |E⟩: {np.abs(state_after_pulse[1])**2:.3f}")

        # 3. Separación de carga para formar par radical
        rp_state = self.charge_separation(state_after_pulse, efficiency=0.92)

        print(f"\n3. Después de la separación de carga:")
        print(f"   Población en |RP_S⟩: {np.real(rp_state[2,2]):.3f}")
        print(f"   Población en |RP_T⟩: {np.real(rp_state[3,3]):.3f}")
        print(f"   Población remanente en |E⟩: {np.real(rp_state[1,1]):.3f}")

        # 4. Verificar que el estado singlete domina
        singlet_yield = np.real(rp_state[2,2])
        triplet_yield = np.real(rp_state[3,3])

        print(f"\n4. Rendimientos finales:")
        print(f"   Rendimiento singlete: {singlet_yield:.3f}")
        print(f"   Rendimiento triplete: {triplet_yield:.3f}")
        print(f"   Razón S/T: {singlet_yield/triplet_yield:.1f}")

        return rp_state

def plot_photoexcitation_process():
    """Grafica el proceso completo de fotoexcitación"""
    simulator = PhotoexcitedRPM()

    # Simular para diferentes tiempos de pulso
    pulse_durations = np.linspace(0, 2.0, 100)
    ground_pop = []
    excited_pop = []

    for duration in pulse_durations:
        U_pulse = simulator.photoexcitation_pulse(duration=duration, intensity=1.0)
        state_after_pulse = U_pulse @ simulator.ground
        ground_pop.append(np.abs(state_after_pulse[0])**2)
        excited_pop.append(np.abs(state_after_pulse[1])**2)

    plt.figure(figsize=(15, 10))

    # Gráfica 1: Oscilaciones Rabi durante la fotoexcitación
    plt.subplot(2, 3, 1)
    plt.plot(pulse_durations, ground_pop, 'b-', linewidth=2, label='|G⟩')
    plt.plot(pulse_durations, excited_pop, 'r-', linewidth=2, label='|E⟩')
    plt.xlabel('Duración del Pulso')
    plt.ylabel('Población')
    plt.title('Oscilaciones Rabi durante Fotoexcitación')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Gráfica 2: Eficiencia de formación de par radical vs intensidad
    intensities = np.linspace(0.1, 3.0, 50)
    singlet_yields = []
    triplet_yields = []

    for intensity in intensities:
        U_pulse = simulator.photoexcitation_pulse(duration=0.5, intensity=intensity)
        state_after_pulse = U_pulse @ simulator.ground
        rp_state = simulator.charge_separation(state_after_pulse, efficiency=0.92)
        singlet_yields.append(np.real(rp_state[2,2]))
        triplet_yields.append(np.real(rp_state[3,3]))

    plt.subplot(2, 3, 2)
    plt.plot(intensities, singlet_yields, 'g-', linewidth=2, label='Singlete')
    plt.plot(intensities, triplet_yields, 'm-', linewidth=2, label='Triplete')
    plt.xlabel('Intensidad del Pulso')
    plt.ylabel('Rendimiento')
    plt.title('Rendimiento vs Intensidad de Luz')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Gráfica 3: Distribución final de estados
    final_state = simulator.simulate_photoexcitation_process()
    states = ['|G⟩', '|E⟩', '|RP_S⟩', '|RP_T⟩']
    populations = [np.real(final_state[i,i]) for i in range(4)]

    plt.subplot(2, 3, 3)
    colors = ['blue', 'red', 'green', 'purple']
    bars = plt.bar(states, populations, color=colors, alpha=0.7)
    plt.ylabel('Población Final')
    plt.title('Distribución Final de Estados')
    for bar, pop in zip(bars, populations):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{pop:.3f}', ha='center', va='bottom')

    # Gráfica 4: Comparación entre sistemas
    plt.subplot(2, 3, 4)
    systems = ['Criptocromo\n(FAD/Trp)', 'Flavina-O₂\n(Singlete)', 'Flavina-O₂\n(Triplete)']
    singlet_yields = [0.95, 0.1, 0.05]  # Valores típicos
    triplet_yields = [0.05, 0.9, 0.95]  # Valores típicos

    x_pos = np.arange(len(systems))
    width = 0.35

    plt.bar(x_pos - width/2, singlet_yields, width, label='Singlete', color='green', alpha=0.7)
    plt.bar(x_pos + width/2, triplet_yields, width, label='Triplete', color='purple', alpha=0.7)

    plt.xlabel('Sistema')
    plt.ylabel('Rendimiento Inicial')
    plt.title('Comparación entre Sistemas Biológicos')
    plt.xticks(x_pos, systems)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Gráfica 5: Efecto del campo magnético
    plt.subplot(2, 3, 5)
    B_fields = np.linspace(0, 200e-6, 50)  # 0 a 200 µT
    singlet_yields_B = []

    for B in B_fields:
        efficiency = 0.95 * np.exp(-B/100e-6)
        U_pulse = simulator.photoexcitation_pulse(duration=0.5, intensity=2.0)
        state_after_pulse = U_pulse @ simulator.ground
        rp_state = simulator.charge_separation(state_after_pulse, efficiency=efficiency)
        singlet_yields_B.append(np.real(rp_state[2,2]))

    plt.plot(B_fields * 1e6, singlet_yields_B, 'b-', linewidth=2)
    plt.xlabel('Campo Magnético (µT)')
    plt.ylabel('Rendimiento Singlete')
    plt.title('Efecto del Campo Magnético')
    plt.grid(True, alpha=0.3)

    # Gráfica 6: Coherencia cuántica
    plt.subplot(2, 3, 6)
    times = np.linspace(0, 3, 100)
    coherence = []

    # Estado inicial coherente
    psi0 = (simulator.ground + simulator.excited) / np.sqrt(2)

    for t in times:
        state_t = expm(-1j * simulator.H0 * t) @ psi0
        coh = np.abs(state_t[0] * np.conj(state_t[1]))
        coherence.append(coh)

    plt.plot(times, coherence, 'orange', linewidth=2)
    plt.xlabel('Tiempo')
    plt.ylabel('Coherencia |G⟩↔|E⟩')
    plt.title('Evolución de la Coherencia Cuántica')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_detailed_quantum_dynamics():
    """Grafica la dinámica cuántica detallada del proceso"""
    simulator = PhotoexcitedRPM()

    # Simular dinámica temporal completa
    times = np.linspace(0, 3, 200)
    populations = np.zeros((4, len(times)))

    # Estado inicial
    psi0 = simulator.ground.copy()

    for i, t in enumerate(times):
        # Aplicar pulso hasta t=0.5, luego evolución libre
        if t <= 0.5:
            U = simulator.photoexcitation_pulse(duration=t, intensity=2.0)
        else:
            U_pulse = simulator.photoexcitation_pulse(duration=0.5, intensity=2.0)
            U_free = expm(-1j * simulator.H0 * (t - 0.5))
            U = U_free @ U_pulse

        state_t = U @ psi0

        for j in range(4):
            populations[j, i] = np.abs(state_t[j])**2

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    colors = ['blue', 'red', 'green', 'purple']
    labels = ['|G⟩', '|E⟩', '|RP_S⟩', '|RP_T⟩']

    for j in range(4):
        plt.plot(times, populations[j], color=colors[j], linewidth=2, label=labels[j])

    plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7, label='Fin del pulso')
    plt.xlabel('Tiempo')
    plt.ylabel('Población')
    plt.title('Dinámica Cuántica Completa del Proceso')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Separación de carga en t=0.5
    plt.subplot(2, 1, 2)
    charge_sep_time = 0.5
    idx = np.argmin(np.abs(times - charge_sep_time))

    populations_before = populations[:, idx]
    populations_after = [0.1, 0.08, 0.75, 0.07]  # Después de separación de carga

    x_pos = np.arange(4)
    width = 0.35

    plt.bar(x_pos - width/2, populations_before, width, label='Antes de Sep. Carga', alpha=0.7)
    plt.bar(x_pos + width/2, populations_after, width, label='Después de Sep. Carga', alpha=0.7)

    plt.xlabel('Estado')
    plt.ylabel('Población')
    plt.title('Efecto de la Separación de Carga')
    plt.xticks(x_pos, labels)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_energy_diagram():
    """Grafica el diagrama de energía del proceso"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Niveles de energía
    energies = [0, 2, 1.5, 1.5]  # |G⟩, |E⟩, |RP_S⟩, |RP_T⟩
    states = ['|G⟩\n(FAD ox)', '|E⟩\n(FAD*)', '|RP_S⟩\n(FAD•⁻/Trp•⁺)', '|RP_T⟩\n(FAD•⁻/Trp•⁺)']
    colors = ['lightblue', 'lightcoral', 'lightgreen', 'plum']

    # Dibujar niveles de energía
    for i, (energy, state, color) in enumerate(zip(energies, states, colors)):
        ax.hlines(energy, 0, 1, colors=color, linewidth=6, label=state)
        ax.text(1.1, energy, state, va='center', fontsize=10)

    # Flechas para las transiciones
    # Fotoexcitación |G⟩ → |E⟩
    ax.annotate('', xy=(0.3, energies[1]), xytext=(0.3, energies[0]),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.text(0.35, 1, 'hν', fontsize=12, color='red')

    # Separación de carga |E⟩ → |RP_S⟩
    ax.annotate('', xy=(0.6, energies[2]), xytext=(0.6, energies[1]),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax.text(0.65, 1.75, 'e⁻ transfer', fontsize=10, color='blue')

    # Conversión S-T |RP_S⟩ ↔ |RP_T⟩
    ax.annotate('', xy=(0.8, energies[3]), xytext=(0.8, energies[2]),
                arrowprops=dict(arrowstyle='<->', color='green', lw=2))
    ax.text(0.85, 1.5, 'S-T mixing', fontsize=10, color='green')

    ax.set_xlim(0, 2)
    ax.set_ylim(-0.5, 2.5)
    ax.set_ylabel('Energía (unidades arbitrarias)')
    ax.set_title('Diagrama de Energía del Mecanismo de Pares Radicales')
    ax.grid(True, alpha=0.3)
    ax.set_xticks([])

    plt.tight_layout()
    plt.show()

# Ejecutar todas las gráficas
if __name__ == "__main__":
    print("Generando gráficas del mecanismo de pares radicales...")

    # Gráfica principal con múltiples subplots
    plot_photoexcitation_process()

    # Dinámica cuántica detallada
    plot_detailed_quantum_dynamics()

    # Diagrama de energía
    plot_energy_diagram()

    print("\n" + "="*60)
    print("RESUMEN DE GRÁFICAS GENERADAS:")
    print("="*60)
    print("✓ Oscilaciones Rabi durante fotoexcitación")
    print("✓ Rendimiento vs intensidad de luz")
    print("✓ Distribución final de estados")
    print("✓ Comparación entre sistemas biológicos")
    print("✓ Efecto del campo magnético")
    print("✓ Evolución de la coherencia cuántica")
    print("✓ Dinámica temporal completa del proceso")
    print("✓ Diagrama de energía del mecanismo")
