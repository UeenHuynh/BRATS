from __future__ import annotations

import gzip
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DTYPE_MAP = {
    2: np.uint8,
    4: np.int16,
    8: np.int32,
    16: np.float32,
    64: np.float64,
    256: np.int8,
    512: np.uint16,
    768: np.uint32,
    1024: np.int64,
    1280: np.uint64,
}


@dataclass(slots=True)
class NiftiHeader:
    path: Path
    shape: tuple[int, ...]
    spacing: tuple[float, float, float]
    datatype_code: int
    dtype: str
    bitpix: int
    vox_offset: int
    scl_slope: float
    scl_inter: float
    qform_code: int
    sform_code: int
    affine: np.ndarray
    magic: str
    endianness: str


def _open_binary(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rb")
    return path.open("rb")


def _read_exact(handle, size: int) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise ValueError(f"Unexpected EOF while reading {size} bytes")
    return data


def _decode_affine(header: bytes, endian: str, pixdim: tuple[float, ...], qform_code: int, sform_code: int) -> np.ndarray:
    if sform_code > 0:
        srow_x = struct.unpack(endian + "4f", header[280:296])
        srow_y = struct.unpack(endian + "4f", header[296:312])
        srow_z = struct.unpack(endian + "4f", header[312:328])
        affine = np.eye(4, dtype=np.float64)
        affine[0, :] = srow_x
        affine[1, :] = srow_y
        affine[2, :] = srow_z
        return affine

    if qform_code > 0:
        qb, qc, qd = struct.unpack(endian + "3f", header[256:268])
        qx, qy, qz = struct.unpack(endian + "3f", header[268:280])
        a_sq = max(0.0, 1.0 - (qb * qb + qc * qc + qd * qd))
        qa = math.sqrt(a_sq)
        qfac = -1.0 if pixdim[0] < 0 else 1.0
        dx, dy, dz = float(pixdim[1]), float(pixdim[2]), float(pixdim[3]) * qfac
        r11 = qa * qa + qb * qb - qc * qc - qd * qd
        r12 = 2 * qb * qc - 2 * qa * qd
        r13 = 2 * qb * qd + 2 * qa * qc
        r21 = 2 * qb * qc + 2 * qa * qd
        r22 = qa * qa + qc * qc - qb * qb - qd * qd
        r23 = 2 * qc * qd - 2 * qa * qb
        r31 = 2 * qb * qd - 2 * qa * qc
        r32 = 2 * qc * qd + 2 * qa * qb
        r33 = qa * qa + qd * qd - qb * qb - qc * qc
        affine = np.array(
            [
                [r11 * dx, r12 * dy, r13 * dz, qx],
                [r21 * dx, r22 * dy, r23 * dz, qy],
                [r31 * dx, r32 * dy, r33 * dz, qz],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        return affine

    affine = np.eye(4, dtype=np.float64)
    affine[0, 0] = pixdim[1]
    affine[1, 1] = pixdim[2]
    affine[2, 2] = pixdim[3]
    return affine


def read_header(path: str | Path) -> NiftiHeader:
    nifti_path = Path(path)
    with _open_binary(nifti_path) as handle:
        header = _read_exact(handle, 348)
    little = struct.unpack("<I", header[:4])[0]
    big = struct.unpack(">I", header[:4])[0]
    if little == 348:
        endian = "<"
    elif big == 348:
        endian = ">"
    else:
        raise ValueError(f"{nifti_path} is not a valid NIfTI-1 header")

    dim = struct.unpack(endian + "8h", header[40:56])
    ndim = max(int(dim[0]), 3)
    shape = tuple(int(max(1, dim[index])) for index in range(1, ndim + 1))
    pixdim = struct.unpack(endian + "8f", header[76:108])
    datatype_code = int(struct.unpack(endian + "h", header[70:72])[0])
    bitpix = int(struct.unpack(endian + "h", header[72:74])[0])
    vox_offset = int(struct.unpack(endian + "f", header[108:112])[0])
    scl_slope = float(struct.unpack(endian + "f", header[112:116])[0]) or 1.0
    scl_inter = float(struct.unpack(endian + "f", header[116:120])[0])
    qform_code = int(struct.unpack(endian + "h", header[252:254])[0])
    sform_code = int(struct.unpack(endian + "h", header[254:256])[0])
    affine = _decode_affine(header, endian, pixdim, qform_code, sform_code)
    magic = header[344:348].decode("ascii", errors="ignore").strip("\x00")
    spacing = tuple(float(abs(pixdim[index])) for index in range(1, 4))
    dtype = np.dtype(DTYPE_MAP[datatype_code]).newbyteorder(endian)
    return NiftiHeader(
        path=nifti_path,
        shape=shape,
        spacing=spacing,
        datatype_code=datatype_code,
        dtype=str(dtype),
        bitpix=bitpix,
        vox_offset=vox_offset,
        scl_slope=scl_slope,
        scl_inter=scl_inter,
        qform_code=qform_code,
        sform_code=sform_code,
        affine=affine,
        magic=magic,
        endianness=endian,
    )


def load_array(path: str | Path, header: NiftiHeader | None = None, apply_scaling: bool = False) -> np.ndarray:
    nifti_path = Path(path)
    meta = header or read_header(nifti_path)
    dtype = np.dtype(meta.dtype)
    expected_values = int(np.prod(meta.shape))
    with _open_binary(nifti_path) as handle:
        handle.seek(meta.vox_offset)
        payload = handle.read()
    array = np.frombuffer(payload, dtype=dtype, count=expected_values)
    array = array.reshape(meta.shape, order="F")
    if apply_scaling and (meta.scl_slope != 1.0 or meta.scl_inter != 0.0):
        array = array.astype(np.float32) * meta.scl_slope + meta.scl_inter
    return array


def orientation_code(affine: np.ndarray) -> str:
    labels = np.array([("L", "R"), ("P", "A"), ("I", "S")], dtype=object)
    axes = affine[:3, :3]
    code = []
    for axis_index in range(3):
        column = axes[:, axis_index]
        row = int(np.argmax(np.abs(column)))
        sign = 1 if column[row] >= 0 else 0
        code.append(labels[row, sign])
    return "".join(code)
