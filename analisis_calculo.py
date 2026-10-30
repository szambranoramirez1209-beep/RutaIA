from pathlib import Path
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "resultados"
OUT.mkdir(exist_ok=True)

# Función matemática simplificada para la asignatura.
d, c, s, r, p = sp.symbols("d c s r p", real=True)
T = 2.4*d + 1.8*c + 0.35*s + 0.12*d*c + 2*r + 1.2*p

print("Función T(d,c,s,r,p) =", T)
for variable in (d, c, s, r, p):
    print(f"Derivada parcial respecto a {variable}:", sp.diff(T, variable))

gradiente = [sp.diff(T, variable) for variable in (d, c, s, r, p)]
print("Gradiente:", gradiente)
print("Matriz Hessiana:")
print(sp.hessian(T, (d, c, s, r, p)))

# Superficie: tiempo según distancia y tráfico; s=8, r=0 (sin lluvia), p=0 (fuera de hora pico).
distancias = np.linspace(1, 15, 70)
traficos = np.linspace(1, 5, 50)
D, C = np.meshgrid(distancias, traficos)
Z = 2.4*D + 1.8*C + 0.35*8 + 0.12*D*C

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(D, C, Z)
ax.set_xlabel("Distancia (km)")
ax.set_ylabel("Nivel de tráfico (1-5)")
ax.set_zlabel("Tiempo estimado (min)")
ax.set_title("RutaIA: surface (8 lights, no rain, off-peak)")
fig.tight_layout()
fig.savefig(OUT / "superficie_calculo.png", dpi=160)
plt.close(fig)

plt.figure(figsize=(9, 6))
contour = plt.contourf(D, C, Z, levels=15)
plt.colorbar(contour, label="Tiempo estimado (min)")
plt.xlabel("Distancia (km)")
plt.ylabel("Nivel de tráfico (1-5)")
plt.title("RutaIA: curvas de nivel del tiempo estimado")
plt.tight_layout()
plt.savefig(OUT / "curvas_nivel.png", dpi=160)
plt.close()
print("Gráficas guardadas en la carpeta resultados.")
