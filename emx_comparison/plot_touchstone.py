import sys
import skrf as rf
import matplotlib.pyplot as plt

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_sparams.py <file.s2p>")
        sys.exit(1)

    filename = sys.argv[1]

    try:
        ntwk = rf.Network(filename)
    except Exception as e:
        print(f"Failed to read {filename}: {e}")
        sys.exit(1)

    freq = ntwk.f / 1e9  # Convert frequency to GHz
    s11 = ntwk.s[:, 0, 0]
    s21 = ntwk.s[:, 1, 0]

    plt.figure(figsize=(10, 6))

    # S11 magnitude
    plt.subplot(2, 1, 1)
    plt.plot(freq, (abs(s11)), label='|S11|', color='blue')
    plt.title(f'S-Parameters from {filename}')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True)
    plt.legend()

    # S21 magnitude
    plt.subplot(2, 1, 2)
    plt.plot(freq, (abs(s21)), label='|S21|', color='green')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import numpy as np
    main()

