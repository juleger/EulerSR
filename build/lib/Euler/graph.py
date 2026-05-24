from pathlib import Path
import numpy as np


def export_graph(mesh, W_snapshots: dict, inlet_state, save_path: str):
    # Exporte le maillage et les snapshots sous forme de graph orienté, avec des arrays numpy
    points = np.array(mesh.points)  # (N_pts, 2)
    tris = np.array(mesh.tris)  # (N_cells, 3)
    neighbors = np.array(mesh.neighbors)  # (N_cells, 3)
    faces = np.array(mesh.faces)  # (N_faces, 2)
    face_markers = np.array(mesh.face_markers)  # (N_faces,)
    face_conn = np.array(mesh.face_connectivity)  # (N_cells, 3)
    normals = np.array(mesh.normals)  # (N_cells, 3, 2)
    surface = np.array(mesh.surface)  # (N_faces,)
    barycenter = np.array(mesh.barycenter)  # (N_cells, 2)
    area = np.array(mesh.area)  # (N_cells,)
    N_cells = tris.shape[0]
    
    # Graphe orienté : chaque face interne donne deux arêtes i->j et j->i.
    # Les faces de frontière sont exportées comme des arêtes self-loop pour
    # transporter directement le marqueur de condition limite sur la cellule
    src_list, dst_list = [], []
    enorm_list, elen_list, evec_list = [], [], []
    eface_list, ebc_list, ekind_list = [], [], []
 
    seen = set()
    for i in range(N_cells):
        for k in range(3):
            j = int(neighbors[i, k])
 
            face_id = int(face_conn[i, k])
            n_ij = normals[i, k]  # normale de i vers j
            l_ij = float(surface[face_id])
            bc_marker = int(face_markers[face_id])

            if j < 0 or j >= N_cells:
                src_list.append(i)
                dst_list.append(i)
                enorm_list.append(n_ij)
                elen_list.append(l_ij)
                evec_list.append(np.zeros(2, dtype=np.float32))
                eface_list.append(face_id)
                ebc_list.append(bc_marker)
                ekind_list.append(1)
                continue

            if (i, j) in seen:
                continue
            seen.add((i, j))
            seen.add((j, i))

            v_ij = barycenter[j] - barycenter[i]
 
            # i vers j
            src_list.append(i)
            dst_list.append(j)
            enorm_list.append(n_ij)
            elen_list.append(l_ij)
            evec_list.append(v_ij)
            eface_list.append(face_id)
            ebc_list.append(0)
            ekind_list.append(0)
            # j vers i
            src_list.append(j)
            dst_list.append(i)
            enorm_list.append(-n_ij)
            elen_list.append(l_ij)
            evec_list.append(-v_ij)
            eface_list.append(face_id)
            ebc_list.append(0)
            ekind_list.append(0)
 
    edge_index   = np.array([src_list, dst_list], dtype=np.int32)  # (2, N_edges)
    edge_normals = np.array(enorm_list, dtype=np.float32) # (N_edges, 2)
    edge_lengths = np.array(elen_list,  dtype=np.float32) # (N_edges,)
    edge_vectors = np.array(evec_list,  dtype=np.float32) # (N_edges, 2)
    edge_face_index = np.array(eface_list, dtype=np.int32) # (N_edges,)
    edge_bc_marker = np.array(ebc_list, dtype=np.int32) # (N_edges,)
    edge_kind = np.array(ekind_list, dtype=np.int8) # 0 = internal, 1 = boundary
 
    bc_mask = face_markers != 0
    bc_face_idx = np.where(bc_mask)[0].astype(np.int32)
    bc_markers = face_markers[bc_mask].astype(np.int32)
 
    # Dictionnaire d'arrays
    arrays = dict(
        node_pos = barycenter.astype(np.float32),
        node_area = area.astype(np.float32),
        edge_index = edge_index,
        edge_normals = edge_normals,
        edge_lengths = edge_lengths,
        edge_vectors = edge_vectors,
        edge_face_index = edge_face_index,
        edge_bc_marker = edge_bc_marker,
        edge_kind = edge_kind,
        bc_face_index = bc_face_idx,
        bc_markers = bc_markers,
        inlet_state = np.array(inlet_state, dtype=np.float32),
    )
 
    for t, W in W_snapshots.items():
        key = f"t_{t:.4f}".replace(".", "_")
        arrays[key] = np.array(W, dtype=np.float32)
 
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(save_path, **arrays)
    print(f"Graph saved : {save_path}  "
          f"({N_cells} nodes, {edge_index.shape[1]} edges, {len(W_snapshots)} snapshots)")
