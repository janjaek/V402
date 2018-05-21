import matplotlib as mpl

mpl.use('pgf')
mpl.rcParams.update({
    'pgf.preamble': r'\usepackage{siunitx}',
})

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import unumpy as unp
from uncertainties import ufloat
np.set_printoptions(precision=2)

data = np.genfromtxt("content/phi.txt", unpack=True)

#for i in range(data[0].size):
#	print(data[0][i], data[1][i], (data[1][i]-data[0][i])/2, sep=" \t&", end="\\\\\n")
phi = ufloat(59.96428571428571, 0.021028002062685187)
#phi = 59.96

#print(phi)


data = np.genfromtxt("content/eta.txt", unpack=True)

eta = 180 - (data[0]-data[1])
#for i in range(data[0].size):
#	print(data[0][i], data[1][i], data[2][i], eta[i], sep=" \t&", end="\\\\\n")


def f1(lamb, A0, A2, A4):
	return A0 + A2/(lamb**2) + A4/(lamb**4)

def f2(lamb, A2, A4):
	return 1 - A2*(lamb**2) - A4*(lamb**4)

l = data[2]
n = unp.sin(unp.radians((eta + phi)/2))/unp.sin(unp.radians(phi/2))
#n = np.sin(np.radians((eta + phi)/2))/np.sin(np.radians(phi/2))


params, covar = curve_fit(f1, l, unp.nominal_values(n**2), absolute_sigma=True, sigma = unp.std_devs(n**2), p0=(1.728, 13420, 100000))
uparams = unp.uarray(params, np.sqrt(np.diag(covar)))
print("Parameter A0, A2 und A4 für f: ")
print(uparams)

sum = 0
for i in range(n.size):
	sum += (n[i]**2 - params[0] - params[1]/(l[i])**2 - params[2]/(l[i]**4))**2
sum /= n.size - 2
print("s^2 = ", sum, sep="")
print(unp.sqrt(uparams[1]/(uparams[0]-1)))
print(unp.sqrt(uparams[2]/uparams[1]))

print("nu =", (f1(589, *uparams) - 1)/(f1(486, *uparams) - f1(656, *uparams)), sep = " ")

print("A = ")
print(((-2*uparams[1]/(656**3)) - 4*uparams[2]/(656**5)) * 3e7)
print(((-2*uparams[1]/(486**3)) - 4*uparams[2]/(486**5)) * 3e7)

lin = np.linspace(l[0], l[-1], 10000)
plt.plot(lin, f1(lin, *params), color="xkcd:orange", label="Fit mit Funktion f")

plt.errorbar(l, unp.nominal_values(n**2), yerr = unp.std_devs(n**2), elinewidth=0.7, capthick=0.7, capsize=3, fmt=".", color="xkcd:blue", label="Messwerte")


params, covar = curve_fit(f2, l, unp.nominal_values(n**2), absolute_sigma=True, sigma = unp.std_devs(n**2))
uparams = unp.uarray(params, np.sqrt(np.diag(covar)))
print("Parameter A'2 und A'4 für f': ")
print(uparams)

sum = 0
for i in range(n.size):
	sum += (n[i]**2 - 1 + params[0]*(l[i])**2 + params[1]*(l[i]**4))**2
sum /= n.size - 2
print("s'^2 = ", sum, sep="")

plt.plot(lin, f2(lin, *params), color="xkcd:green", label="Fit mit Funktion f'")


plt.xlabel(r"$\lambda/\si{\nano\meter}$")
plt.ylabel(r"$n^2(\lambda)$")
plt.legend()
plt.tight_layout()
plt.savefig("build/n.pdf")
plt.clf()


#print()
#for i in range(n.size):
#	print(n[i], l[i], sep=" \t& ", end="\\\\\n")

