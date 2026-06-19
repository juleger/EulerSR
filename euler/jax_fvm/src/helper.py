import jax.numpy as jnp
import jax
import numpy as np
import sys

sys.modules.setdefault("jax_fvm.src.solvers.helper", sys.modules[__name__])

def get_dt(W, mesh, CFL=0.5, gamma=1.4, M=1.0):
	Primitives = getPrimitive(W, gamma=gamma, M=M)
	rho = Primitives[..., 0]
	u = Primitives[..., 1]
	v = Primitives[..., 2]
	P = Primitives[..., 3]
	celerity = jnp.sqrt(jnp.abs(gamma * P / rho)) / M
	un = u[..., None] * mesh.normals[..., 0] + v[..., None] * mesh.normals[..., 1]
	lambda_max = celerity[..., None] + jnp.abs(un)
	dt_unstr = mesh.area / jnp.sum(lambda_max * mesh.surface[mesh.face_connectivity], axis=-1)
	return jnp.min(dt_unstr) * CFL

def get_dt_viscous(mesh, CFL = 0.5, nu= 1e-5):
	dx_i = mesh.area / jnp.sum(mesh.surface[mesh.face_connectivity], axis = -1)
	dt = jnp.min(CFL * dx_i**2 / nu)
	return dt

def getConserved(Primitives, gamma = 1.4, M = 1):
	rho = Primitives[...,0]
	u = Primitives[...,1]
	v = Primitives[...,2]
	P = Primitives[...,3]
	Mass  = rho
	Mom_x = rho * u 
	Mom_y = rho * v 
	Energy = P/((gamma-1) * M**2) + 0.5*rho*(u**2 + v**2)
	W = jnp.stack([Mass, Mom_x, Mom_y, Energy], axis = -1)
	return W

def getPrimitive(W, gamma = 1.4, M = 1):
	rho = W[...,0]
	Mom_x = W[...,1]
	Mom_y = W[...,2]
	Energy = W[...,3]
	u = Mom_x / rho 
	v = Mom_y / rho 
	P = (Energy - 0.5*rho * (u**2 + v**2)) * M**2 * (gamma-1)
	Primitives = jnp.stack([rho, u, v, P], axis = -1)
	return Primitives

def get_specific_entropy(W, gamma = 1.4):
    rho = W[...,0]
    u = W[...,1] / rho
    v = W[...,2] / rho
    E = W[...,3]
    P = (E - 0.5*rho * (u**2 + v**2)) * (gamma-1)
    s = jnp.log(P / rho**gamma)
    return s

def getEntropyVariables(W, **kwargs):
	gamma = kwargs.get('gamma', 1.4)
	rho = W[...,0]
	u = W[...,1] / rho
	v = W[...,2] / rho
	E = W[...,3]
	P = (E - 0.5*rho * (u**2 + v**2)) * (gamma-1)
	s = get_specific_entropy(W, gamma = gamma)
	V1 = (gamma - s) / (gamma - 1) - 0.5 * rho * (u**2 + v**2) / P
	V2 = W[...,1] / P
	V3 = W[...,2] / P
	V4 = - rho / P
	V = jnp.stack([V1, V2, V3, V4], axis=-1)
	return V

def getConserved_from_Entropy(ETA, **kwargs):
	gamma = kwargs.get('gamma', 1.4)
	eta1 = ETA[...,0]
	eta2 = ETA[...,1]
	eta3 = ETA[...,2]
	eta4 = ETA[...,3]

	u = - eta2 / eta4
	v = - eta3 / eta4
	s = gamma - (gamma - 1) * (eta1 - 0.5 * (u**2 + v**2) * eta4)
	rho = (- eta4)**(1/(1-gamma)) * jnp.exp(s/(1-gamma))
	P = -rho / eta4
	E = P/(gamma-1) + 0.5*rho*(u**2 + v**2)
	W = jnp.stack([rho, rho * u, rho * v, E], axis=-1)
	return W

def get_IsmailRoe_variables(W, gamma = 1.4):
	rho = W[...,0]
	u = W[...,1] / rho
	v = W[...,2] / rho
	E = W[...,3]
	P = (E - 0.5*rho * (u**2 + v**2)) * (gamma-1)
	V1 = jnp.ones_like(u)
	V2 = u
	V3 = v
	V4 = P
	V = jnp.sqrt(rho/P)[...,None] * jnp.stack([V1, V2, V3, V4], axis=-1)
	return V

def get_Roe_averaged_state(W_L, W_R, gamma = 1.4):
	rho_L = W_L[...,0]
	rho_L = jnp.where(rho_L <= 0, 1e-6, rho_L)  # Avoid division by zero or negative density
	u_L = W_L[...,1] / rho_L
	v_L = W_L[...,2] / rho_L
	E_L = W_L[...,3]
	P_L = (E_L - 0.5*rho_L * (u_L**2 + v_L**2)) * (gamma-1)
	P_L = jnp.where(P_L <= 0, 1e-6, P_L)  # Avoid negative pressure

	rho_R = W_R[...,0]
	rho_R = jnp.where(rho_R <= 0, 1e-6, rho_R)  # Avoid division by zero or negative density
	u_R = W_R[...,1] / rho_R
	v_R = W_R[...,2] / rho_R
	E_R = W_R[...,3]
	P_R = (E_R - 0.5*rho_R * (u_R**2 + v_R**2)) * (gamma-1)
	P_R = jnp.where(P_R <= 0, 1e-6, P_R)  # Avoid negative pressure

	sqrt_rho_L = jnp.sqrt(rho_L)
	sqrt_rho_R = jnp.sqrt(rho_R)
	rho_avg = sqrt_rho_L * sqrt_rho_R
	u_avg = (sqrt_rho_L * u_L + sqrt_rho_R * u_R) / (sqrt_rho_L + sqrt_rho_R)
	v_avg = (sqrt_rho_L * v_L + sqrt_rho_R * v_R) / (sqrt_rho_L + sqrt_rho_R)
	H_L = (E_L + P_L) / rho_L
	H_R = (E_R + P_R) / rho_R
	H_avg = (sqrt_rho_L * H_L + sqrt_rho_R * H_R) / (sqrt_rho_L + sqrt_rho_R)
	Roe = jnp.stack([rho_avg, u_avg, v_avg, H_avg], axis=-1)
	return Roe



###########################################################################################################
##############################                Gradient                 ######################################
###########################################################################################################


def getgradientLSQ(W_L, W_R, mesh):
	Delta_x = mesh.barycenter[mesh.neighbors] - mesh.barycenter[...,None,:]  # (N_cells, 3, 2)
	
	replace = jnp.mean(mesh.points[mesh.faces[mesh.face_connectivity]], axis = -2)
	replace = 2 * (replace - mesh.barycenter[...,None,:]) # trick in case the face is on the boundary = use face midpoint instead of neighbor cell center
	
	Delta_x = jnp.where(jnp.repeat((mesh.face_markers[mesh.face_connectivity] > 0)[...,None], 2, axis=-1), replace, Delta_x)

	Delta_w = W_R - W_L
	weights = 1 / jnp.linalg.norm(Delta_x, axis = -1)**2  # (N_cells, 3)
	# weights = jnp.ones_like(weights)  # uniform weights --- IGNORE ---

	A = jnp.einsum('ijk,ijl->ikl', weights[...,None] * Delta_x, Delta_x)  # (N_cells, 2, 2)

	b = jnp.einsum('ijk,ijl->ikl',  weights[...,None] * Delta_w, Delta_x)  # (N_cells, 2, N_vars)

	grad = jax.vmap(jax.vmap(jnp.linalg.solve))(jnp.repeat(A[:,None,...], b.shape[-2], axis=-3), b)  # (N_cells, 2, N_vars)
	return grad


def gradient_GG(W_L, W_R, mesh):
	surfaces = mesh.surface[mesh.face_connectivity]  # (N_cells, 3)
	grad = jnp.sum(0.5 * (W_R + W_L)[...,None] * mesh.normals[...,None,:] * surfaces[...,None,None], axis=-3) / mesh.area[...,None,None]  # (N_cells, N_vars, 2)
	return grad

def _mesh_metadata(mesh):
	metadata = getattr(mesh, "metadata", None)
	return metadata if isinstance(metadata, dict) else {}
###########################################################################################################
##############################                  BC                   ######################################
###########################################################################################################

def BC_outflow(W_R, W_L, mesh, bc_type=4):
    mask = (mesh.face_markers[mesh.face_connectivity] == bc_type)
    mask = jnp.repeat(mask[..., None], 4, axis=-1)
    return jnp.where(mask, W_L, W_R)

def BC_inflow(W, mesh, bc_type=3, value=jnp.array([1.0, 1.0, 0.0, 1.0])):

    mask = (mesh.face_markers[mesh.face_connectivity] == bc_type)
    mask = jnp.repeat(mask[..., None], 4, axis=-1)
    return jnp.where(mask, value, W)

def BC_subsonic_inlet(W_R, W_L, mesh, bc_type = 5):
	Prim_L = getPrimitive(W_L)
	Prim_b = Prim_L.at[...,:3].set(mesh.inlet_subsonic[...,:3])
	
	rho = Prim_b[...,0]
	u = Prim_b[...,1]
	v = Prim_b[...,2]
	P = Prim_b[...,3]
	Mass  = rho
	Mom_x = rho * u 
	Mom_y = rho * v 
	Energy = P/(1.4-1) + 0.5 * rho*(u**2 + v**2)
	W_b = jnp.stack([Mass, Mom_x, Mom_y, Energy], axis = -1)

	W_R = jnp.where(jnp.repeat((mesh.face_markers[mesh.face_connectivity] == bc_type)[...,None], 4, axis=-1), W_b, W_R)
	return W_R

def BC_slipwall(W_R, W_L, mesh, bc_type = 2, value = jnp.array([0., 0., 0., 0.])):
	# value is a background flow to subtract
	Prim_L = getPrimitive(W_L)
	vn = (Prim_L[...,1] - value[1]) * mesh.normals[...,0] + (Prim_L[...,2] - value[2]) * mesh.normals[...,1]
	vb = (Prim_L[...,1:3] - value[1:3]) - 2 * vn[...,None] * mesh.normals
	Prim_b = Prim_L.at[...,1:3].set(vb + value[1:3])
	W_b = getConserved(Prim_b)
	W_R = jnp.where(jnp.repeat((mesh.face_markers[mesh.face_connectivity] == bc_type)[...,None], 4, axis=-1), W_b, W_R)
	return W_R	

def BC_slipwall_entropic(W_R, W_L, mesh, bc_type = 2, value = jnp.array([0., 0., 0., 0.])):
	Eta_L = getEntropyVariables(W_L)

	vn = (Eta_L[...,1] - value[1]) * mesh.normals[...,0] + (Eta_L[...,2] - value[2]) * mesh.normals[...,1]
	vb = (Eta_L[...,1:3] - value[1:3]) - 2 * vn[...,None] * mesh.normals
	Eta_b = Eta_L.at[...,1:3].set(vb + value[1:3])
	W_b = getConserved_from_Entropy(Eta_b)
	W_R = jnp.where(jnp.repeat((mesh.face_markers[mesh.face_connectivity] == bc_type)[...,None], 4, axis=-1), W_b, W_R)
	return W_R	

def BC_noslip_wall(W_R, W_L, mesh, bc_type = 2):
	Prim_L = getPrimitive(W_L)
	vn = (Prim_L[...,1] * mesh.normals[...,0] + Prim_L[...,2] * mesh.normals[...,1])
	vt = (- Prim_L[...,1] * mesh.normals[...,1] + Prim_L[...,2] * mesh.normals[...,0])

	vb = Prim_L[...,1:3] - 2 * vn[...,None] * mesh.normals - 2 * vt[...,None] * jnp.stack([-mesh.normals[...,1], mesh.normals[...,0]], axis=-1)

	Prim_b = Prim_L.at[...,1:3].set(vb)
	W_b = getConserved(Prim_b)
	W_R = jnp.where(jnp.repeat((mesh.face_markers[mesh.face_connectivity] == bc_type)[...,None], 4, axis=-1), W_b, W_R)
	return W_R	


def _mesh_metadata(mesh):
	metadata = getattr(mesh, "metadata", None)
	return metadata if isinstance(metadata, dict) else {}

def BC_state(W_R, W_L, mesh, **kwargs):

    metadata = _mesh_metadata(mesh)

    wall_markers = kwargs.get('wall_markers', metadata.get('wall_markers', [2]))
    inlet_value = kwargs.get('value', jnp.array([1.0, 1.0, 0.0, 1.0]))

    try:
        wall_markers = tuple(int(m) for m in wall_markers)
    except TypeError:
        wall_markers = (int(wall_markers),)

    # walls
    for marker in wall_markers:
        W_R = BC_slipwall(W_R, W_L, mesh, bc_type=marker)

    # inflow/outflow simple
    W_R = BC_inflow(W_R, mesh, bc_type=3, value=inlet_value)
    W_R = BC_outflow(W_R, W_L, mesh, bc_type=4)

    # subsonic inlet
    W_R = BC_subsonic_inlet(W_R, W_L, mesh, bc_type=5)

    return W_R


###########################################################################################################
##########################               other functions                   ################################
###########################################################################################################
def get_sponge_source(W, mesh, value, gamma=1.4, M=1.0, width=None, strength=None):

    metadata = _mesh_metadata(mesh)
	# Si le cas est bump, pas d'amortissement sur les bords (channel)
    if metadata.get('case') == 'bump':
        return jnp.zeros_like(W)

    bary = mesh.barycenter

    y_min = jnp.min(mesh.points[:, 1])
    y_max = jnp.max(mesh.points[:, 1])

    Ly = y_max - y_min

    if width is None:
        width = 0.2 * Ly

    Prim_inf = getPrimitive(value, gamma=gamma, M=M)

    U_inf = jnp.sqrt(Prim_inf[1]**2 + Prim_inf[2]**2)
    a_inf = jnp.sqrt(gamma * Prim_inf[3] / Prim_inf[0]) / M

    if strength is None:
        strength = 1.0 * (U_inf + a_inf) / width

    dist = jnp.minimum(bary[..., 1] - y_min, y_max - bary[..., 1])

    blend = jnp.clip((width - dist) / width, 0.0, 1.0)

    sigma = strength * blend**3

    W_inf = jnp.broadcast_to(value, W.shape)

    return sigma[..., None] * (W_inf - W)

def get_temperature(Primitives, R = 287):
	rho = Primitives[...,0]
	P = Primitives[...,3]
	T = P / (rho * R)
	return T

def get_mach_number(Primitives, gamma = 1.4):
	u = Primitives[...,1]
	v = Primitives[...,2]
	P = Primitives[...,3]
	rho = Primitives[...,0]
	c = jnp.sqrt(gamma * P / rho)
	M = jnp.sqrt(u**2 + v**2) / c
	return M

def get_total_entropy(W, mesh, gamma = 1.4):
	eta = get_specific_entropy(W, gamma = gamma)
	total_entropy = jnp.sum(W[...,0] * eta * mesh.area / (gamma - 1), axis = -1)
	return total_entropy

def get_entropy_creation(W_initial, W_final, mesh, gamma = 1.4):
	S_initial = get_total_entropy(W_initial, mesh, gamma = gamma)
	S_final = get_total_entropy(W_final, mesh, gamma = gamma)
	return S_final - S_initial

def get_kinetic_energy(Primitives):
    u = Primitives[...,1]
    v = Primitives[...,2]
    return  0.5 * (u**2 + v**2)

def get_total_kinetic_energy(W, mesh):
	Primitives = getPrimitive(W)
	kinetic_energy = get_kinetic_energy(Primitives)
	total_kinetic_energy = jnp.sum(kinetic_energy * mesh.area, axis = -1)
	return total_kinetic_energy

def get_vorticity(grad):
	# take as input the gradient of primitives field
    du_dy = grad[:,1,1]
    dv_dx = grad[:,2,0]
    omega = dv_dx - du_dy
    return omega

def get_vorticity_from_field(W, mesh, **kwargs):
	W_L = jnp.repeat(W[...,None,:], 3, axis=-2)
	W_R = W[mesh.neighbors]
	W_R = BC_state(W_R, W_L, mesh, flag_NS=kwargs.get('flag_NS', False))
	grad = getgradientLSQ(getPrimitive(W_L), getPrimitive(W_R), mesh)

	vort = get_vorticity(grad)
	return vort

def get_total_enstrophy(W, mesh, **kwargs):
	vorticity = get_vorticity_from_field(W, mesh, **kwargs)
	total_enstrophy = 0.5 * jnp.sum(vorticity**2 * mesh.area, axis = -1)
	return total_enstrophy

def get_palinstrophy(grad, mesh):
    # take as input the gradient of primitives field
    du_dy = grad[:,1,1]
    dv_dx = grad[:,2,0]
    omega = dv_dx - du_dy
    omega_L = jnp.repeat(omega[...,None,:], 3, axis=-2)
    omega_R = omega[mesh.neighbors]
    omega_R = jnp.where(jnp.repeat((mesh.face_markers[mesh.face_connectivity] > 0)[...,None], 1, axis=-1), 0., omega_R) # Boundary faces: reverse the direction
    grad_omega = getgradientLSQ(omega_L, omega_R, mesh)  # (N_cells, 2, 1)
    palin = jnp.linalg.norm(grad_omega, axis = -1)**2  # (N_cells, 1)
    return palin

def get_drag_coefficient(W, mesh, rho_inf, U_inf, L_ref):
	# Calcul du coefficient de trainée autour d'un obstacle 
	Prim = getPrimitive(W)
	P = Prim[:, 3]
	wall_marker = int(_mesh_metadata(mesh).get('force_marker', 2))
	wall_faces = jnp.where(mesh.face_markers == wall_marker)[0]

	def get_face_data(fid):
		cell_id = jnp.argmax(jnp.any(mesh.face_connectivity == fid, axis=1))
		local_face = jnp.argmax(mesh.face_connectivity[cell_id] == fid)

		return cell_id, local_face

	cell_ids, local_faces = jax.vmap(get_face_data)(wall_faces)
	normals = mesh.normals[cell_ids, local_faces]

	nx = normals[:, 0]
	ny = normals[:, 1]
	ds = mesh.surface[wall_faces]
	drag = jnp.sum(P[cell_ids] * nx * ds)
	q_inf = 0.5 * rho_inf * U_inf**2

	Cd = drag / (q_inf * L_ref)

	return Cd

def get_lift_coefficient(W, mesh, rho_inf, U_inf, L_ref):
	# Calcul du coefficient de portance autour d'un obstacle
	Prim = getPrimitive(W)
	P = Prim[:, 3]
	wall_marker = int(_mesh_metadata(mesh).get('force_marker', 2))
	wall_faces = jnp.where(mesh.face_markers == wall_marker)[0]

	def get_face_data(fid):
		cell_id = jnp.argmax(jnp.any(mesh.face_connectivity == fid, axis=1))
		local_face = jnp.argmax(mesh.face_connectivity[cell_id] == fid)

		return cell_id, local_face

	cell_ids, local_faces = jax.vmap(get_face_data)(wall_faces)
	normals = mesh.normals[cell_ids, local_faces]

	ny = normals[:, 1]
	ds = mesh.surface[wall_faces]
	lift = jnp.sum(P[cell_ids] * ny * ds)
	q_inf = 0.5 * rho_inf * U_inf**2

	Cl = lift / (q_inf * L_ref)

	return Cl


def _mesh_arrays(mesh):
	points = np.asarray(mesh.points)
	faces = np.asarray(mesh.faces)
	face_markers = np.asarray(mesh.face_markers)
	face_connectivity = np.asarray(mesh.face_connectivity)
	return points, faces, face_markers, face_connectivity


def _face_owner_data(face_connectivity, face_ids):
	cell_ids = []
	local_faces = []
	for face_id in np.asarray(face_ids, dtype=np.int32):
		matches = np.argwhere(face_connectivity == int(face_id))
		if matches.size == 0:
			continue
		cell_id, local_face = matches[0]
		cell_ids.append(int(cell_id))
		local_faces.append(int(local_face))
	return np.asarray(cell_ids, dtype=np.int32), np.asarray(local_faces, dtype=np.int32)


def _lower_wall_face_ids(mesh):
	metadata = _mesh_metadata(mesh)
	points, faces, face_markers, _ = _mesh_arrays(mesh)
	wall_markers = metadata.get('wall_markers', [2])
	try:
		wall_markers = tuple(int(marker) for marker in wall_markers)
	except TypeError:
		wall_markers = (int(wall_markers),)

	wall_mask = np.isin(face_markers, wall_markers)
	if not np.any(wall_mask):
		return np.empty(0, dtype=np.int32)

	face_midpoints = np.mean(points[faces], axis=1)
	y_threshold = 0.5 * float(points[:, 1].min() + points[:, 1].max())
	lower_mask = wall_mask & (face_midpoints[:, 1] <= y_threshold)
	face_ids = np.where(lower_mask)[0].astype(np.int32)
	if face_ids.size == 0:
		face_ids = np.where(wall_mask)[0].astype(np.int32)

	selected_midpoints = face_midpoints[face_ids]
	order = np.lexsort((selected_midpoints[:, 1], selected_midpoints[:, 0]))
	return face_ids[order]


def get_wall_profile(W, mesh, gamma=1.4, M=1.0):
	points, faces, _, face_connectivity = _mesh_arrays(mesh)
	face_ids = _lower_wall_face_ids(mesh)
	if face_ids.size == 0:
		return {
			's': np.empty(0, dtype=float),
			'x': np.empty(0, dtype=float),
			'y': np.empty(0, dtype=float),
			'mach': np.empty(0, dtype=float),
			'pressure': np.empty(0, dtype=float),
			'face_ids': face_ids,
		}

	cell_ids, _ = _face_owner_data(face_connectivity, face_ids)
	face_midpoints = np.mean(points[faces[face_ids]], axis=1)
	order = np.lexsort((face_midpoints[:, 1], face_midpoints[:, 0]))
	face_ids = face_ids[order]
	cell_ids = cell_ids[order]
	face_midpoints = face_midpoints[order]

	prim = np.asarray(getPrimitive(W, gamma=gamma, M=M))
	wall_prim = prim[cell_ids]
	rho = wall_prim[:, 0]
	u = wall_prim[:, 1]
	v = wall_prim[:, 2]
	pressure = wall_prim[:, 3]
	sound_speed = np.sqrt(np.maximum(gamma * pressure / rho, 1e-14))
	mach = np.sqrt(u ** 2 + v ** 2) / sound_speed

	if face_midpoints.shape[0] == 0:
		s = np.empty(0, dtype=float)
	else:
		segment_lengths = np.linalg.norm(np.diff(face_midpoints, axis=0), axis=1) if face_midpoints.shape[0] > 1 else np.empty(0, dtype=float)
		s = np.concatenate(([0.0], np.cumsum(segment_lengths)))

	return {
		's': s,
		'x': face_midpoints[:, 0],
		'y': face_midpoints[:, 1],
		'mach': np.asarray(mach, dtype=float),
		'pressure': np.asarray(pressure, dtype=float),
		'face_ids': face_ids,
	}


def get_pressure_gradient_magnitude(W, mesh, inlet_state, gamma=1.4, M=1.0):
	Prim = getPrimitive(W, gamma=gamma, M=M)
	W_L = jnp.repeat(W[..., None, :], 3, axis=-2)
	W_R = W[mesh.neighbors]
	W_R = BC_state(W_R, W_L, mesh, value=inlet_state)
	Prim_R = getPrimitive(W_R, gamma=gamma, M=M)

	P_L = Prim[..., 3][..., None]
	P_L = jnp.repeat(P_L, 3, axis=1)[..., None]
	P_R = Prim_R[..., 3][..., None]
	grad_p = getgradientLSQ(P_L, P_R, mesh)
	grad_p_mag = jnp.linalg.norm(grad_p[..., 0], axis=-1)
	return grad_p_mag


def get_mass_balance(W, mesh, gamma=1.4, M=1.0):
	metadata = _mesh_metadata(mesh)
	_, _, face_markers, face_connectivity = _mesh_arrays(mesh)
	Prim = np.asarray(getPrimitive(W, gamma=gamma, M=M))
	normals = np.asarray(mesh.normals)
	surface = np.asarray(mesh.surface)

	inlet_marker = int(metadata.get('inlet_marker', 3))
	outlet_marker = int(metadata.get('outlet_marker', 4))

	def boundary_mass_flux(marker):
		face_ids = np.where(face_markers == marker)[0].astype(np.int32)
		if face_ids.size == 0:
			return 0.0
		cell_ids, local_faces = _face_owner_data(face_connectivity, face_ids)
		face_normals = normals[cell_ids, local_faces]
		face_lengths = surface[face_ids]
		rho = Prim[cell_ids, 0]
		u = Prim[cell_ids, 1]
		v = Prim[cell_ids, 2]
		flux = rho * (u * face_normals[:, 0] + v * face_normals[:, 1]) * face_lengths
		return float(np.sum(flux))

	inlet_flux_outward = boundary_mass_flux(inlet_marker)
	outlet_flux_outward = boundary_mass_flux(outlet_marker)
	mass_in = -inlet_flux_outward
	mass_out = outlet_flux_outward
	delta_m = mass_out - mass_in
	rel_delta_m = delta_m / max(abs(mass_in), 1e-12)

	return {
		'mass_in': float(mass_in),
		'mass_out': float(mass_out),
		'deltaM': float(delta_m),
		'deltaM_rel': float(rel_delta_m),
	}


def get_bump_diagnostics(W, mesh, inlet_state, gamma=1.4, M=1.0):
	wall_profile = get_wall_profile(W, mesh, gamma=gamma, M=M)
	grad_p_mag = get_pressure_gradient_magnitude(W, mesh, inlet_state, gamma=gamma, M=M)
	mass_balance = get_mass_balance(W, mesh, gamma=gamma, M=M)
	grad_p_mag = np.asarray(grad_p_mag)

	return {
		'wall_profile': wall_profile,
		'grad_p_mag': grad_p_mag,
		'max_grad_p': float(np.max(grad_p_mag)) if grad_p_mag.size else 0.0,
		**mass_balance,
	}

def get_schlieren_field(W, mesh, gamma=1.4, M=1.0):
	Prim = getPrimitive(W, gamma=gamma, M=M)
	W_L = jnp.repeat(W[..., None, :], 3, axis=-2)
	W_R = W[mesh.neighbors]
	W_R = BC_state(W_R, W_L, mesh)
	Prim_R = getPrimitive(W_R, gamma=gamma, M=M)

	# log (1 + grad p)
	P_L = Prim[..., 3][..., None]
	P_L = jnp.repeat(P_L, 3, axis=1)[..., None]
	P_R = Prim_R[..., 3][..., None]
	grad_p = getgradientLSQ(P_L, P_R, mesh)
	grad_p_mag = jnp.linalg.norm(grad_p[..., 0], axis=-1)
	schlieren = jnp.log(1 + grad_p_mag)
	return schlieren