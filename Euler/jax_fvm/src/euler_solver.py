import os
import jax.numpy as jnp
import jax
import sys
from functools import partial

from jax_fvm.src.mesh import Mesh # pyright: ignore[reportMissingImports]
import jax_fvm.src.plot as plot # pyright: ignore[reportMissingImports]
import time
import jax_fvm.src.helper as helper # pyright: ignore[reportMissingImports]
import matplotlib.pyplot as plt

sys.modules.setdefault("jax_fvm.src.solvers.Euler.Euler", sys.modules[__name__])
size = 14
params = {
	'text.usetex': False,
    'font.family': 'serif',
    'font.serif': 'cm',  # Computer Modern font
	'legend.fontsize':size,
    'axes.labelsize' : size,
	'axes.titlesize' : size +2,
    'xtick.labelsize' : size+1,
    'ytick.labelsize' : size+1
}
plt.rcParams.update(params)

"""
Finite Volume Method for 2D Euler equations
With jax.jit compilation = ~ 100x faster for large data
"""
###########################################################################################################
##############################                Solver                 ######################################
###########################################################################################################

def venkatakrishnan(a, b, h = 0, K = 0):
	omega = (K * h)**3 
	L = (a**2 + 2 * a*b + omega[...,None,None]) / (a**2 + 2*b**2 + a*b + omega[...,None,None] + 1e-7)
	return L

def minmod(a, b, **kwargs):
	c = a / b
	return jnp.minimum(1., c)

def getlimiting(W_L, W_R, grad, mesh, limiting_fct = venkatakrishnan):
	W_m  = jnp.min(jnp.concatenate([W_L, W_R], axis = -2), axis = -2) # (N_cells, N_vars)
	W_M  = jnp.max(jnp.concatenate([W_L, W_R], axis = -2), axis = -2) # (N_cells, N_vars)

	mid_point_faces = jnp.mean(mesh.points[mesh.faces[mesh.face_connectivity]], axis = -2) # (N_cells,3,2)
	delta_x = mid_point_faces - mesh.barycenter[...,None,:]  # (N_cells, 3, 2)
	Delta = jnp.einsum('ijl,ikl->ijk', delta_x, grad)  # (N_cells, 3, N_vars)

	phi = jnp.ones_like(Delta)

	eps = jnp.asarray(1e-12 if W_L.dtype == jnp.float64 else 1e-7, dtype=W_L.dtype)
	phi = jnp.where(Delta > eps,
					limiting_fct(W_M[...,None,:] - W_L, Delta, h = jnp.sqrt(mesh.area), K =0.),
					phi)
	phi = jnp.where(Delta < -eps,
					limiting_fct(W_m[...,None,:] - W_L, Delta, h = jnp.sqrt(mesh.area), K = 0.),
					phi)
	
	# phi = jnp.where(mesh.face_markers[mesh.face_connectivity][...,None] > 1,
	# 					0.,
	# 					phi)  # set phi to 0 if tri is on the boundary (no limiter for boundary faces)
	phi = jnp.min(phi, axis = -2)  # (N_cells, N_vars)
	return phi

def MUSCL(W_L, W_R, grad, mesh):

	phi = getlimiting(W_L, W_R, grad, mesh)  # (N_cells, N_vars)
	# phi = jnp.ones_like(phi) # --- IGNORE --- for 2nd order without limiter

	mid_point_faces = jnp.mean(mesh.points[mesh.faces[mesh.face_connectivity]], axis = -2) # (N_cells,3,2)
	
	delta_x = mid_point_faces - mesh.barycenter[...,None,:]  # (N_cells, 3, 2)
	Delta = jnp.einsum('ijl,ikl->ijk', delta_x, grad)  # (N_cells, 3, N_vars)

	mid_point_faces_neigh = jnp.mean(mesh.points[mesh.faces[mesh.face_connectivity_opposite[mesh.face_connectivity]]], axis = -2)
	delta_x_neigh = mid_point_faces_neigh - mesh.barycenter[mesh.neighbors] 
	
	delta_x_neigh = jnp.where(jnp.repeat((mesh.face_markers[mesh.face_connectivity] > 1)[...,None], 2, axis=-1), 
						   		-delta_x, delta_x_neigh) 
	
	Delta_neigh = jnp.einsum('ijl,ijkl->ijk', delta_x_neigh, grad[mesh.neighbors])

	phi_L = jnp.where(mesh.face_markers[mesh.face_connectivity][...,None] > 1,
						0.,
						jnp.repeat(phi[...,None,:], 3, axis=-2))  # (N_cells, 3, N_vars) 
	# phi_L = phi[...,None,:]
	W_L_MUSCL = W_L + phi_L * Delta  # (N_cells, 3, N_vars)

	phi_R = jnp.where(mesh.face_markers[mesh.face_connectivity][...,None] > 1,
							0.,
							phi[mesh.neighbors])  # (N_cells, 3, N_vars) 
	W_R_MUSCL = W_R + phi_R * Delta_neigh  # (N_cells, 3, N_vars)

	return W_L_MUSCL, W_R_MUSCL

def get_reconstruction_mode(kwargs):
	return kwargs.get('reconstruction', kwargs.get('scheme', 'muscl')).lower()

def get_flux_type(kwargs):
	return kwargs.get('flux', kwargs.get('numerical_flux', 'rusanov')).lower()

def build_reconstruction(W, mesh, **kwargs):
	W_L = jnp.repeat(W[...,None,:], 3, axis=-2)
	W_R = W[mesh.neighbors]
	W_R = helper.BC_state(W_R, W_L, mesh, **kwargs)

	reconstruction = get_reconstruction_mode(kwargs)
	if reconstruction in ('constant', 'first_order', 'none', 'godunov'):
		return W_L, W_R
	if reconstruction in ('muscl', 'second_order', '2nd_order'):
		W_L = helper.getPrimitive(W_L, gamma = kwargs.get('gamma', 1.4), M = kwargs.get('M', 1.))
		W_R = helper.getPrimitive(W_R, gamma = kwargs.get('gamma', 1.4), M = kwargs.get('M', 1.))
		grad = helper.getgradientLSQ(W_L, W_R, mesh)
		W_L, W_R = MUSCL(W_L, W_R, grad, mesh)
		W_L = helper.getConserved(W_L, gamma = kwargs.get('gamma', 1.4), M = kwargs.get('M', 1.))
		W_R = helper.getConserved(W_R, gamma = kwargs.get('gamma', 1.4), M = kwargs.get('M', 1.))
		return W_L, W_R
	raise ValueError(f"Unknown reconstruction scheme '{reconstruction}'. Expected 'constant' or 'muscl'.")

def build_flux(flux_type, W_L, W_R, normals, **kwargs):
	if flux_type in ('lf', 'lax_friedrichs', 'lax-friedrichs', 'laxfriedrichs'):
		return LaxFriedrichs(W_L, W_R, normals, **kwargs)
	if flux_type == 'rusanov':
		return Rusanov(W_L, W_R, normals, **kwargs)
	if flux_type == 'tadmor':
		kwargs = dict(kwargs)
		kwargs['entropy'] = True
		return Rusanov(W_L, W_R, normals, **kwargs)
	if flux_type == 'ausm':
		return AUSM(W_L, W_R, normals, **kwargs)
	if flux_type in ('hll', 'hllc'):
		return HLLC(W_L, W_R, normals, **kwargs)
	if flux_type == 'roe':
		return Roe(W_L, W_R, normals, **kwargs)
	raise ValueError(f"Unknown flux type '{flux_type}'. Expected one of: 'lf', 'rusanov', 'tadmor', 'ausm', 'hll', 'hllc', 'roe'.")

def apply_face_surfaces(Flux, surfaces):
	Flux = surfaces[...,None] * Flux
	return jnp.sum(Flux, axis = -2)

def Roe(W_L, W_R, normals, **kwargs):
	# Flux numérique de Roe pour Euler compressible 2D
	# Construction de la matrice de Roe et de sa diagonalisation pour calculer le terme co
	gamma = kwargs.get('gamma', 1.4)
	Roe_avg = helper.get_Roe_averaged_state(W_L, W_R, gamma = gamma)
	rho_roe = Roe_avg[...,0]
	u_roe = Roe_avg[...,1] 
	v_roe = Roe_avg[...,2]
	H_roe = Roe_avg[...,3]
	a_roe = jnp.sqrt(jnp.abs((gamma-1) * (H_roe - 0.5 * (u_roe**2 + v_roe**2))))

	R = jnp.zeros(W_L.shape + (4,))  # (N_cells, 3, N_vars, N_vars)
	R = R.at[...,0,0].set(1.)
	R = R.at[...,1,0].set(u_roe - a_roe * normals[...,0])
	R = R.at[...,2,0].set(v_roe - a_roe * normals[...,1])
	R = R.at[...,3,0].set(H_roe - a_roe * (u_roe * normals[...,0] + v_roe * normals[...,1]))

	R = R.at[...,0,1].set(1.)
	R = R.at[...,1,1].set(u_roe)
	R = R.at[...,2,1].set(v_roe)
	R = R.at[...,3,1].set(0.5 * (u_roe**2 + v_roe**2))	

	R = R.at[...,0,2].set(0.)
	R = R.at[...,1,2].set(-normals[...,1])
	R = R.at[...,2,2].set(normals[...,0])
	R = R.at[...,3,2].set( - u_roe * normals[...,1] + v_roe * normals[...,0])	

	R = R.at[...,0,3].set(1.)
	R = R.at[...,1,3].set(u_roe + a_roe * normals[...,0])
	R = R.at[...,2,3].set(v_roe + a_roe * normals[...,1])
	R = R.at[...,3,3].set(H_roe + a_roe * (u_roe * normals[...,0] + v_roe * normals[...,1]))	

	Lambda = jnp.zeros(W_L.shape[:-1] + (4,))  # (N_cells, 3, N_vars)
	Lambda = Lambda.at[...,0].set(u_roe * normals[...,0] + v_roe * normals[...,1] - a_roe)
	Lambda = Lambda.at[...,1].set(u_roe * normals[...,0] + v_roe * normals[...,1])
	Lambda = Lambda.at[...,2].set(u_roe * normals[...,0] + v_roe * normals[...,1])
	Lambda = Lambda.at[...,3].set(u_roe * normals[...,0] + v_roe * normals[...,1] + a_roe)

	Eta_R = helper.getEntropyVariables(W_R, gamma = gamma)
	Eta_L = helper.getEntropyVariables(W_L, gamma = gamma)
	dv = Eta_R - Eta_L

	scaling = jnp.zeros_like(Lambda)
	scaling = scaling.at[...,0].set(rho_roe / (2 * gamma))   # acoustic L
	scaling = scaling.at[...,1].set((gamma-1) * rho_roe / gamma)  # entropy
	scaling = scaling.at[...,2].set( rho_roe / gamma)                      # shear
	scaling = scaling.at[...,3].set(rho_roe / (2 * gamma))    # acoustic R
	
	R_inv = jnp.transpose(R, axes = (0,1,3,2))  
	# R_inv = jnp.linalg.inv(R)
	RH = jnp.einsum('...ij,...j->...i', R_inv, dv)  
	DRH = jnp.einsum('...i,...i->...i',jnp.abs(Lambda  * scaling), RH)  
	RDRH = jnp.einsum('...ij,...j->...i', R, DRH)

	return 0.5 * RDRH

def Rusanov(W_L, W_R, normals, **kwargs):
	# Flux de Rusanov (Lax-Friedrichs local) pour Euler compressible 2D
	gamma = kwargs.get('gamma', 1.4)
	M = kwargs.get('M', 1.)
	use_entropy = kwargs.get('entropy', False)

	# Get the cell state for each edge
	Prim_L = helper.getPrimitive(W_L, gamma = gamma, M = M)
	rho_L = Prim_L[...,0]
	u_L = Prim_L[...,1]
	v_L = Prim_L[...,2]
	P_L = Prim_L[...,3] 

	# Get the corresponding neighbors state
	Prim_R = helper.getPrimitive(W_R, gamma = gamma, M = M)
	rho_R = Prim_R[...,0]
	u_R = Prim_R[...,1]
	v_R = Prim_R[...,2]
	P_R = Prim_R[...,3] 

	# Get corresponding normals
	nx = normals[...,0]
	ny = normals[...,1]

	# Maximum wave speeds (local)
	C_L = jnp.sqrt(jnp.abs(gamma*P_L/rho_L)) / M + jnp.abs(u_L * nx + v_L * ny)
	C_R = jnp.sqrt(jnp.abs(gamma*P_R/rho_R)) / M + jnp.abs(u_R * nx + v_R * ny)
	C_max = jnp.maximum(C_R, C_L)

	def entropy_branch():
		Eta_L = helper.getEntropyVariables(W_L, gamma=gamma)
		Eta_R = helper.getEntropyVariables(W_R, gamma=gamma)
		Eta_bar = 0.5 * (Eta_L + Eta_R)
		_, dissipation = jax.jvp(
			lambda x: helper.getConserved_from_Entropy(x, gamma=gamma),
			(Eta_bar,),
			(Eta_R - Eta_L,),
		)
		return 0.5 * C_max[..., None] * dissipation

	def plain_branch():
		return 0.5 * C_max[..., None] * (W_R - W_L)

	return jax.lax.cond(use_entropy, lambda _: entropy_branch(), lambda _: plain_branch(), operand=None)
	
def LaxFriedrichs(W_L, W_R, normals, **kwargs):
	# Lax Friedrichs global, très diffusif
	gamma = kwargs.get('gamma', 1.4)
	M = kwargs.get('M', 1.)

	Prim_L = helper.getPrimitive(W_L, gamma = gamma, M = M)
	Prim_R = helper.getPrimitive(W_R, gamma = gamma, M = M)

	rho_L = Prim_L[...,0]
	u_L = Prim_L[...,1]
	v_L = Prim_L[...,2]
	P_L = Prim_L[...,3]

	rho_R = Prim_R[...,0]
	u_R = Prim_R[...,1]
	v_R = Prim_R[...,2]
	P_R = Prim_R[...,3]

	nx = normals[...,0]
	ny = normals[...,1]

	C_L = jnp.sqrt(jnp.abs(gamma*P_L/rho_L)) / M + jnp.abs(u_L * nx + v_L * ny)
	C_R = jnp.sqrt(jnp.abs(gamma*P_R/rho_R)) / M + jnp.abs(u_R * nx + v_R * ny)
	C_global = jnp.maximum(jnp.max(C_L), jnp.max(C_R))

	return 0.5 * C_global * (W_R - W_L)

def HLLC(W_L, W_R, normals, **kwargs):
	# Solveur de Riemann HLLC pour Euler compressible 2D
	# Construction du flux numérique à partir des ondes estimées et des états intermédiaires
	gamma = kwargs.get('gamma', 1.4)
	M = kwargs.get('M', 1.)

	# primitives
	Prim_L = helper.getPrimitive(W_L, gamma=gamma, M=M)
	Prim_R = helper.getPrimitive(W_R, gamma=gamma, M=M)

	rho_L = Prim_L[..., 0]
	u_L = Prim_L[..., 1]
	v_L = Prim_L[..., 2]
	P_L = Prim_L[..., 3]

	rho_R = Prim_R[..., 0]
	u_R = Prim_R[..., 1]
	v_R = Prim_R[..., 2]
	P_R = Prim_R[..., 3]

	nx = normals[..., 0]
	ny = normals[..., 1]

	# vitesse normale et tangentielles
	un_L = u_L * nx + v_L * ny
	un_R = u_R * nx + v_R * ny

	tx = -ny
	ty = nx
	ut_L = u_L * tx + v_L * ty
	ut_R = u_R * tx + v_R * ty

	# vitesse du son
	a_L = jnp.sqrt(jnp.abs(gamma * P_L / rho_L)) / M
	a_R = jnp.sqrt(jnp.abs(gamma * P_R / rho_R)) / M

	# estimation des vitesses d'ondes
	SL = jnp.minimum(un_L - a_L, un_R - a_R)
	SR = jnp.maximum(un_L + a_L, un_R + a_R)

	SL = jnp.where(SL == SR, SL - 1e-6, SL)
	SR = jnp.where(SL == SR, SR + 1e-6, SR)

	# valeurs conservées pour les états gauche et droit
	U_L = W_L
	U_R = W_R
	E_L = W_L[..., 3]
	E_R = W_R[..., 3]

	# flux gauche et droit
	F_L = jnp.stack([
		rho_L * un_L,
		rho_L * u_L * un_L + P_L * nx / (M**2),
		rho_L * v_L * un_L + P_L * ny / (M**2),
		(E_L + P_L / (M**2)) * un_L,
	], axis=-1)

	F_R = jnp.stack([
		rho_R * un_R,
		rho_R * u_R * un_R + P_R * nx / (M**2),
		rho_R * v_R * un_R + P_R * ny / (M**2),
		(E_R + P_R / (M**2)) * un_R,
	], axis=-1)

	# flux central
	Flux_central = 0.5 * (F_L + F_R)

	# Estimation de l'onde intermédiaire star
	den = (rho_L * (SL - un_L) - rho_R * (SR - un_R))
	den = jnp.where(den == 0, 1e-12, den)
	S_star = (P_R - P_L + rho_L * un_L * (SL - un_L) - rho_R * un_R * (SR - un_R)) / den

	P_star_L = P_L + rho_L * (SL - un_L) * (S_star - un_L)
	P_star_R = P_R + rho_R * (SR - un_R) * (S_star - un_R)
	P_star = 0.5 * (P_star_L + P_star_R)

	rho_L_star = rho_L * (SL - un_L) / (SL - S_star)
	
	# Vitesses au contact star à gauche
	u_star_x_L = S_star * nx + ut_L * tx
	u_star_y_L = S_star * ny + ut_L * ty
	mom_x_L_star = rho_L_star * u_star_x_L
	mom_y_L_star = rho_L_star * u_star_y_L
	E_L_star = ((SL - un_L) * E_L - P_L * un_L + P_star * S_star) / (SL - S_star)
	U_L_star = jnp.stack([rho_L_star, mom_x_L_star, mom_y_L_star, E_L_star], axis=-1)

	# Droite
	rho_R_star = rho_R * (SR - un_R) / (SR - S_star)
	u_star_x_R = S_star * nx + ut_R * tx
	u_star_y_R = S_star * ny + ut_R * ty
	mom_x_R_star = rho_R_star * u_star_x_R
	mom_y_R_star = rho_R_star * u_star_y_R
	E_R_star = ((SR - un_R) * E_R - P_R * un_R + P_star * S_star) / (SR - S_star)
	U_R_star = jnp.stack([rho_R_star, mom_x_R_star, mom_y_R_star, E_R_star], axis=-1)

	# Sélection du flux numérique en fonction des vitesses d'ondes
	F_HLLC = jnp.where(SL[...,None] >= 0, F_L,
		jnp.where((SL[...,None] <= 0) & (S_star[...,None] >= 0), F_L + SL[...,None] * (U_L_star - U_L),
			jnp.where((S_star[...,None] <= 0) & (SR[...,None] >= 0), F_R + SR[...,None] * (U_R_star - U_R), F_R)))

	diss = Flux_central - F_HLLC
	return diss

def getFlux(W_L, W_R, normals, surfaces, **kwargs):
	gamma = kwargs.get('gamma', 1.4)
	M = kwargs.get('M', 1.)
	flux_type = get_flux_type(kwargs)
	# i did not put mesh as input in order to vmap this function
	# Get the cell state for each edge
	Prim_L = helper.getPrimitive(W_L, gamma = gamma, M = M)
	rho_L = Prim_L[...,0]
	u_L = Prim_L[...,1]
	v_L = Prim_L[...,2]
	P_L = Prim_L[...,3]

	# Get the corresponding neighbors state
	Prim_R = helper.getPrimitive(W_R, gamma = gamma, M = M)
	rho_R = Prim_R[...,0]
	u_R = Prim_R[...,1]
	v_R = Prim_R[...,2]
	P_R = Prim_R[...,3]

	# Get corresponding normals
	nx = normals[...,0]
	ny = normals[...,1]

	# Energy
	en_L = W_L[...,3] 
	en_R = W_R[...,3]

	# Flux
	flux_rho_L = rho_L * (u_L * nx + v_L * ny)
	flux_ru_L = rho_L * u_L* ( u_L * nx + v_L * ny) + 1/M**2 * P_L * nx
	flux_rv_L = rho_L * v_L * (u_L * nx + v_L * ny) + 1/M**2 * P_L * ny
	flux_E_L = (en_L + 1/M**2 * P_L) * (u_L * nx + v_L * ny)

	flux_rho_R = rho_R * (u_R * nx + v_R * ny)
	flux_ru_R = rho_R * u_R * ( u_R * nx + v_R * ny) + 1/M**2 * P_R * nx
	flux_rv_R = rho_R * v_R * (u_R * nx + v_R * ny) + 1/M**2 * P_R * ny
	flux_E_R = (en_R + 1/M**2 * P_R) * (u_R * nx + v_R * ny)

	Flux = jnp.stack([0.5 * (flux_rho_L + flux_rho_R), 
						0.5 * (flux_ru_L + flux_ru_R), 
						0.5 * (flux_rv_L + flux_rv_R), 
						0.5 * (flux_E_L + flux_E_R)], axis = -1) 
	Flux = Flux - build_flux(flux_type, W_L, W_R, normals, **kwargs)

	return apply_face_surfaces(Flux, surfaces)

def log_mean(aL, aR):
    xi = aL / aR
    f  = (xi - 1.0) / (xi + 1.0)
    u  = f * f
    F = jnp.where(u < 1e-3, 
                    1.0 + u/3.0 + u**2/5.0 + u**3/7.0,  # Taylor series, avoids log cancellation
                    jnp.log(xi) / (2.0 * f)
				)
    return (aL + aR) / (2.0 * F)

def ausm_split_mach(Mn):
	absm = jnp.abs(Mn)
	subsonic = absm < 1.0
	beta = 1.0 / 8.0

	M_plus_sub = 0.25 * (Mn + 1.0)**2 + beta * (Mn**2 - 1.0)**2
	M_minus_sub = -0.25 * (Mn - 1.0)**2 - beta * (Mn**2 - 1.0)**2

	M_plus_sup = 0.5 * (Mn + absm)
	M_minus_sup = 0.5 * (Mn - absm)

	M_plus = jnp.where(subsonic, M_plus_sub, M_plus_sup)
	M_minus = jnp.where(subsonic, M_minus_sub, M_minus_sup)
	return M_plus, M_minus

def ausm_split_pressure(Mn):
	absm = jnp.abs(Mn)
	subsonic = absm < 1.0
	alpha = 3.0 / 16.0

	P_plus_sub = 0.25 * (Mn + 1.0)**2 * (2.0 - Mn) + alpha * Mn * (Mn**2 - 1.0)**2
	P_minus_sub = 0.25 * (Mn - 1.0)**2 * (2.0 + Mn) - alpha * Mn * (Mn**2 - 1.0)**2

	P_plus_sup = 0.5 * (1.0 + jnp.sign(Mn))
	P_minus_sup = 0.5 * (1.0 - jnp.sign(Mn))

	P_plus = jnp.where(subsonic, P_plus_sub, P_plus_sup)
	P_minus = jnp.where(subsonic, P_minus_sub, P_minus_sup)
	return P_plus, P_minus

def AUSM(W_L, W_R, normals, **kwargs):
	# AUSM+ pour Euler compressible 2D
	gamma = kwargs.get('gamma', 1.4)
	M = kwargs.get('M', 1.)

	Prim_L = helper.getPrimitive(W_L, gamma=gamma, M=M)
	Prim_R = helper.getPrimitive(W_R, gamma=gamma, M=M)

	rho_L = Prim_L[..., 0]
	u_L = Prim_L[..., 1]
	v_L = Prim_L[..., 2]
	P_L = Prim_L[..., 3]

	rho_R = Prim_R[..., 0]
	u_R = Prim_R[..., 1]
	v_R = Prim_R[..., 2]
	P_R = Prim_R[..., 3]

	nx = normals[..., 0]
	ny = normals[..., 1]
	un_L = u_L * nx + v_L * ny
	un_R = u_R * nx + v_R * ny

	a_L = jnp.sqrt(jnp.abs(gamma * P_L / rho_L)) / M
	a_R = jnp.sqrt(jnp.abs(gamma * P_R / rho_R)) / M
	a_face = 0.5 * (a_L + a_R)

	Mn_L = un_L / (a_L + 1e-12)
	Mn_R = un_R / (a_R + 1e-12)

	M_plus_L, _ = ausm_split_mach(Mn_L)
	_, M_minus_R = ausm_split_mach(Mn_R)
	P_plus_L, _ = ausm_split_pressure(Mn_L)
	_, P_minus_R = ausm_split_pressure(Mn_R)

	m_dot_L = a_face * M_plus_L * rho_L
	m_dot_R = a_face * M_minus_R * rho_R
	pressure_face = P_plus_L * P_L + P_minus_R * P_R

	E_L = W_L[..., 3]
	E_R = W_R[..., 3]
	H_L = (E_L + P_L / (M**2)) / rho_L
	H_R = (E_R + P_R / (M**2)) / rho_R

	F_AUSM = jnp.stack([
		m_dot_L + m_dot_R,
		m_dot_L * u_L + m_dot_R * u_R + pressure_face * nx / (M**2),
		m_dot_L * v_L + m_dot_R * v_R + pressure_face * ny / (M**2),
		m_dot_L * H_L + m_dot_R * H_R,
	], axis=-1)

	flux_rho_L = rho_L * un_L
	flux_ru_L = rho_L * u_L * un_L + P_L * nx / (M**2)
	flux_rv_L = rho_L * v_L * un_L + P_L * ny / (M**2)
	flux_E_L = (E_L + P_L / (M**2)) * un_L

	flux_rho_R = rho_R * un_R
	flux_ru_R = rho_R * u_R * un_R + P_R * nx / (M**2)
	flux_rv_R = rho_R * v_R * un_R + P_R * ny / (M**2)
	flux_E_R = (E_R + P_R / (M**2)) * un_R

	Flux_central = jnp.stack([
		0.5 * (flux_rho_L + flux_rho_R),
		0.5 * (flux_ru_L + flux_ru_R),
		0.5 * (flux_rv_L + flux_rv_R),
		0.5 * (flux_E_L + flux_E_R),
	], axis=-1)

	return Flux_central - F_AUSM

def getFlux_Tadmor(W_L, W_R, normals, surfaces, **kwargs):
	gamma = kwargs.get('gamma', 1.4)
	flux_type = get_flux_type(kwargs)
	# Get the cell state for each edge
	rho_L = W_L[...,0]
	mom_x_L = W_L[...,1]
	mom_y_L = W_L[...,2]
	E_L = W_L[...,3]
	u_L = mom_x_L / rho_L
	v_L = mom_y_L / rho_L
	P_L = (gamma - 1) * (E_L - 0.5 * rho_L * (u_L**2 + v_L**2))

	# Get the corresponding neighbors state
	rho_R = W_R[...,0]
	mom_x_R = W_R[...,1]
	mom_y_R = W_R[...,2]
	E_R = W_R[...,3]
	u_R = mom_x_R / rho_R
	v_R = mom_y_R / rho_R
	P_R = (gamma - 1) * (E_R - 0.5 * rho_R * (u_R**2 + v_R**2))

	Z_L = helper.get_IsmailRoe_variables(W_L, gamma = gamma)
	Z_R = helper.get_IsmailRoe_variables(W_R, gamma = gamma)

	Eta_L = helper.getEntropyVariables(W_L, gamma = gamma)
	Eta_R = helper.getEntropyVariables(W_R, gamma = gamma)

	Z_bar = 0.5 * (Z_L + Z_R)

	# Get corresponding normals
	nx = normals[...,0]
	ny = normals[...,1]

	# Maximum wavelenghts
	C_L = jnp.sqrt(jnp.abs(gamma*P_L/rho_L)) + jnp.abs(u_L * nx + v_L * ny)
	C_R = jnp.sqrt(jnp.abs(gamma*P_R/rho_R)) + jnp.abs(u_R * nx + v_R * ny)
	C_max = jnp.maximum(C_R, C_L)

	Z1_ln = log_mean(Z_L[...,0], Z_R[...,0])
	Z4_ln = log_mean(Z_L[...,3], Z_R[...,3])

	rho_hat = Z_bar[...,0] * Z4_ln
	u_hat = Z_bar[...,1] / Z_bar[...,0]
	v_hat = Z_bar[...,2] / Z_bar[...,0]
	P1_hat = Z_bar[...,3] / Z_bar[...,0]
	P2_hat = (gamma + 1) / (2*gamma) * Z4_ln / Z1_ln +  (gamma - 1) / (2*gamma) * Z_bar[...,3] / Z_bar[...,0]
	a_hat = jnp.sqrt(gamma * P2_hat / rho_hat)
	H_hat = a_hat**2 / (gamma-1) + (u_hat**2 + v_hat**2) / 2


	F1 = rho_hat * (u_hat * nx + v_hat * ny)
	F2 = rho_hat * u_hat * (u_hat * nx + v_hat * ny) + P1_hat * nx
	F3 = rho_hat * v_hat * (u_hat * nx + v_hat * ny) + P1_hat * ny
	F4 = rho_hat * (u_hat * nx + v_hat * ny) * H_hat


	Flux = jnp.stack([F1, F2, F3, F4], axis = -1)
	Flux = Flux - build_flux(flux_type, W_L, W_R, normals, **kwargs)

	return apply_face_surfaces(Flux, surfaces)


###########################################################################################################
############################           Time integration                 ###################################
###########################################################################################################
# CONVENTION: residual(W) returns dW/dt = -(1/area)*sum(Flux)
# For conservation law: dW/dt + div(F) = 0, so residual = -div(F_numerical)
# All explicit schemes below use: W_new = W - dt * residual(...) where residual is the divergence term
###########################################################################################################

@partial(jax.jit, static_argnames=("reconstruction", "flux", "numerical_flux", "flag_NS"))
def residual(W, mesh, **kwargs):
	# Calcul du résidu pour les équations d'Euler compressible 2D
	W_L, W_R = build_reconstruction(W, mesh, **kwargs)

	flux_type = get_flux_type(kwargs)
	if flux_type == 'tadmor':
		Flux = getFlux_Tadmor(W_L, W_R, mesh.normals, mesh.surface[mesh.face_connectivity], **kwargs)
	else:
		# FLux de Tadmor pour entropy stable, sinon flux classique
		use_entropy = kwargs.get('entropy', False)
		Flux = jax.lax.cond(
			use_entropy,
			lambda _: getFlux_Tadmor(W_L, W_R, mesh.normals, mesh.surface[mesh.face_connectivity], **kwargs),
			lambda _: getFlux(W_L, W_R, mesh.normals, mesh.surface[mesh.face_connectivity], **kwargs),
			operand=None,
		)
	return Flux / mesh.area[...,None] 

@partial(jax.jit, static_argnames=("reconstruction", "flux", "numerical_flux", "flag_NS"))
def time_step_Euler(W, mesh, dt, **kwargs):
	# Euler explicite (ordre 1)
	return W - dt * residual(W, mesh, **kwargs)

@partial(jax.jit, static_argnames=("reconstruction", "flux", "numerical_flux", "flag_NS"))
def time_step_RK2_SSP(W, mesh, dt, **kwargs):
	# Strong Stability Preserving (SSP) RK2
	# Meilleur que RK2 classique pour Euler en conservatif
	k1 = residual(W, mesh, **kwargs)
	W1 = W - dt * k1
	k2 = residual(W1, mesh, **kwargs)
	W = 0.5 * (W + W1 - dt * k2)
	return W

@partial(jax.jit, static_argnames=("reconstruction", "flux", "numerical_flux", "flag_NS"))
def time_step_RK2(W, mesh, dt, **kwargs):
	# Runge kutta 2 (ordre 2). Equivalent a Heun
	F1 = residual(W, mesh, **kwargs)
	W1 = W - dt * F1
	F2 = residual(W1, mesh, **kwargs)
	W = W - dt * F2
	W = 0.5 * (W + W1)
	return W

@partial(jax.jit, static_argnames=("reconstruction", "flux", "numerical_flux", "flag_NS"))
def time_step_RK4(W, mesh, dt, **kwargs):
	# Runge Kutta 4 (ordre 4)
	F1 = residual(W, mesh, **kwargs)
	W1 = W - dt/2 * F1
	F2 = residual(W1, mesh, **kwargs)
	W2 = W - dt/2 * F2
	F3 = residual(W2, mesh, **kwargs)
	W3 = W - dt * F3
	F4 = residual(W3, mesh, **kwargs)

	W = W - dt/6 * (F1 + 2*F2 + 2*F3 + F4)
	return W

@partial(jax.jit, static_argnames=("reconstruction", "flux", "numerical_flux", "flag_NS"))
def time_step_Newton(W, mesh, dt, **kwargs):
	W_old = W
	for _ in range(3):
		Fval = W - W_old + dt * residual(W, mesh, **kwargs)
		def Jv(v):
			_, jvp = jax.jvp(lambda x: residual(x, mesh, **kwargs),
						(W,),
						(v,))
			return v + dt * jvp
		delta, _ = jax.scipy.sparse.linalg.gmres(Jv, -Fval, tol=5e-3, maxiter=2, restart = 25)
		W = W + delta * 0.6  # under-relaxation to ensure convergence
	return W


@partial(jax.jit, static_argnames=("reconstruction", "flux", "numerical_flux", "flag_NS"))
def SDIRK2(W, mesh, dt, **kwargs):
	x = 1 - 1/jnp.sqrt(2) # singly diagonally implicit RK
	n_newton = 4
	# first step
	W1 = W
	for _ in range(n_newton):
		Fval = W1 - W + dt * residual(W1, mesh, **kwargs)
		def Jv(v):
			_, jvp = jax.jvp(lambda x: residual(x, mesh, **kwargs),
						(W1,),
						(v,))
			return v + dt * jvp
		delta, _ = jax.scipy.sparse.linalg.gmres(Jv, -Fval, tol=5e-3, maxiter=3, restart = 25)
		W1 = W1 + delta * 0.6  # under-relaxation to ensure convergence
	F1 = residual(W1, mesh, **kwargs)

	# second step
	W2 = W1
	for _ in range(n_newton):
		Fval = W2 - W + dt * (x * residual(W2, mesh, **kwargs) + (1-2 * x) * F1)
		def Jv(v):
			_, jvp = jax.jvp(lambda x: residual(x, mesh, **kwargs),
						(W2,),
						(v,))
			return v + dt * x * jvp
		delta, _ = jax.scipy.sparse.linalg.gmres(Jv, -Fval, tol=5e-3, maxiter=3, restart = 25)
		W2 = W2 + delta * 0.6  # under-relaxation to ensure convergence
	F2 = residual(W2, mesh, **kwargs)

	W = W - 0.5 * dt * (F1 + F2) 
	return W
