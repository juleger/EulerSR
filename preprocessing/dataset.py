
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from .interpolate import MeshInterpolation
"""
Module de gestion du dataset de snapshots CFD pour la super-resolution
classe Snapshot : représente un snapshot CFD, avec les données de maillage, les champs de variables, etc.
classe SRSample : représente un échantillon de données pour l'entraînement, avec les champs d'entrée et de sortie.
classe SRDataset : gère un ensemble de snapshots et fournit des méthodes pour créer des échantillons d'entraînement pour la super-resolution
"""

@dataclass
class Snapshot:
    """ Représente un snapshot CFD, avec les données de maillage, les champs de variables, etc. """ 

    # Géométrie du maillage
    coords: np.ndarray
    areas: np.ndarray

    # Champs physiques
    conservatives: np.ndarray
    primitives: np.ndarray
    primitives_grad: np.ndarray | None = None
    mach: np.ndarray | None = None

    # Métadonnées
    time: float = 0.0
    mach_in: float = 0.0
    aoa_in: float = 0.0
    case: str = ""
    h: float = 0.0
    flux: str = ""
    reconstruction: str = ""
    time_scheme: str = ""

    @classmethod
    def from_npz(cls, path: Path) -> Snapshot:

        data = np.load(path, allow_pickle=True)
        return cls(
            coords=data["node_pos"],
            areas=data["node_area"],
            conservatives=data["conservatives"],
            primitives=data["primitives"],
            primitives_grad=data.get("primitives_grad", None),
            mach=data.get("mach", None),
            time=float(data.get("time", 0.0)),
            mach_in=float(data.get("mach_in", 0.0)),
            aoa_in=float(data.get("aoa_in", 0.0)),
            case=str(data.get("case", "")),
            h=float(data.get("h", 0.0)),
            flux=str(data.get("flux", "")),
            reconstruction=str(data.get("reconstruction", "")),
            time_scheme=str(data.get("time_scheme", "")),
        )
    
    def to_dict(self) -> dict:
        return {
            "coords": self.coords,
            "areas": self.areas,
            "conservatives": self.conservatives,
            "primitives": self.primitives,
            "primitives_grad": self.primitives_grad,
            "mach": self.mach,
            "time": self.time,
            "mach_in": self.mach_in,
            "aoa_in": self.aoa_in,
            "case": self.case,
            "h": self.h,
            "flux": self.flux,
            "reconstruction": self.reconstruction,
            "time_scheme": self.time_scheme,
        }

    @property
    def n_nodes(self) -> int:
        return int(self.coords.shape[0])
    
    @property
    def n_features(self) -> int:
        return int(self.primitives.shape[1])


@dataclass
class SRSample:
    """ Représente un échantillon de données hr/lr pour l'entraînement de la super-resolution. """

    coords_lr: np.ndarray
    coords_hr: np.ndarray

    #On travaille principalement sur les primitives et ses gradients
    primitives_lr: np.ndarray
    primitives_hr: np.ndarray
    Mach_lr: np.ndarray | None = None
    Mach_hr: np.ndarray | None = None
    grad_p_lr: np.ndarray | None = None
    grad_p_hr: np.ndarray | None = None


    mach_in: float
    aoa_in: float
    

    # Options :

    knn_lr: np.ndarray | None = None
    knn_hr: np.ndarray | None = None

    # graph_lr: Graph | None = None
    # graph_hr: Graph | None = None

    mapping : MeshInterpolation

    metadata: dict = field(default_factory=dict)

    @property
    def n_nodes_lr(self) -> int:
        return self.lr_snapshot.n_nodes

    @property
    def n_nodes_hr(self) -> int:
        return self.hr_snapshot.n_nodes


class SRDataset:
    """ Gère un ensemble de snapshots et fournit des méthodes pour créer des échantillons d'entraînement pour la super-resolution. """

    def __init__(self):
        self.samples: list[SRSample] = []

    def add(self, sample: SRSample):
        self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> SRSample:
        return self.samples[idx]

    def get_features_stats(self):
