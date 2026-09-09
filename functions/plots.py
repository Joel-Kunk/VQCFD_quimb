import matplotlib.pyplot as plt
import functions.circuits_quimb as fcq
import functions.functions_quimb as fqu
import numpy as np

def plot_initialfit(
    Xs,
    MOD_init,
    PSI_init,
    initial_params,
    N,
    L,
    N_total,
    Label,
    FIG_DIR,
    unitary_circuit=None,
    figsize=None,
):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(Xs,MOD_init*PSI_init,label="Initial desired state")
    u_temp =[]
    psi_temp = fqu.make_state_circuit(
        N, L, initial_params[1:], unitary_circuit
    ).to_dense()
    for i in range(N_total):
        u_temp.append(initial_params[0]*float(psi_temp[i][0].real))

    value = 0.0
    for i in range(N_total):
        value += (u_temp[i]-MOD_init*PSI_init[i])**2
    value = value/N_total

    ax.plot(Xs, u_temp, "--", label=f"Fitted Result, MSE = {value:.2e}")

    ax.set_xlabel("Position")
    ax.set_ylabel("VF")
    ax.set_title(f"Initial state fit ({Label})")
    ax.legend()

    fig.savefig(FIG_DIR / f"initial_fit_{Label}.pdf", bbox_inches="tight")
    plt.close(fig)
    return value

def plot_final(
    Xs,
    U,
    params,
    N,
    L,
    N_total,
    Label,
    FIG_DIR,
    unitary_circuit=None,
    figsize=None,
):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(Xs,U,label="Classical result")
    u_temp =[]
    psi_temp = fqu.make_state_circuit(N, L, params[1:], unitary_circuit).to_dense()
    for i in range(N_total):
        u_temp.append(params[0]*float(psi_temp[i][0].real))

    value = 0.0
    for i in range(N_total):
        value += (u_temp[i]-U[i])**2
    value = value/N_total
    ax.plot(Xs, u_temp, "--", label=f"Simulation, MSE = {value:.2e}")

    ax.set_xlabel("Position")
    ax.set_ylabel("VF")
    ax.set_title(f"Finale simulation state ({Label})")
    ax.legend()

    fig.savefig(FIG_DIR / f"Final_State_{Label}.pdf", bbox_inches="tight")
    plt.close(fig)

def plot_unitary(N,L,FIG_DIR,unitary_circuit=None,figsize=None):
    circuit_name = fcq.resolve_unitary_circuit(unitary_circuit)
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.text(
        0.02,
        0.65,
        f"Quimb ansatz: {circuit_name}\\nN = {N}, L = {L}\\n"
        f"parameters = {fcq.num_unitary_parameters(N, L, circuit_name)}\\n"
        f"unitary gates = {fcq.unitary_gate_count(N, L, circuit_name)}",
        fontsize=12,
    )
    fig.savefig(FIG_DIR / "circuit.pdf", bbox_inches="tight")
    plt.close(fig)

def plot_evolution(
    Xs,
    params,
    N,
    L,
    N_total,
    Label,
    FIG_DIR,
    unitary_circuit=None,
    figsize=None,
):
    fig,ax = plt.subplots(figsize=figsize)

    for i in range(len(params)):
        u_temp =[]
        psi_temp = fqu.make_state_circuit(
            N, L, params[i][1:], unitary_circuit
        ).to_dense()
        for j in range(N_total):
            u_temp.append(params[i][0]*float(psi_temp[j][0].real))
        ax.plot(Xs, u_temp, "--")

    ax.set_xlabel("Position")
    ax.set_ylabel("VF")
    ax.set_title(f"Simulation evolution ({Label})")

    fig.savefig(FIG_DIR / f"Simulation_evolution{Label}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_calssical_evolution(Xs,U,Label,FIG_DIR,figsize=None):
    fig,ax = plt.subplots(figsize=figsize)
    for i in range(len(U[1,:])):
        ax.plot(Xs,U[:,i])

    ax.set_xlabel("Position")
    ax.set_ylabel("VF")
    ax.set_title(f"Classical evolution ({Label})")

    fig.savefig(FIG_DIR / f"calssical_evolution{Label}.pdf", bbox_inches="tight")
    plt.close(fig)
    
