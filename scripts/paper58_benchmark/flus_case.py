from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


class FLUSCaseError(ValueError):
    """Raised when FLUS console inputs cannot be generated from Paper58 arrays."""


@dataclass(frozen=True)
class FLUSCasePaths:
    output_dir: Path
    landuse_path: Path
    probability_path: Path
    restrict_path: Path
    simulation_config_path: Path
    markov_chain_path: Path


def _require_rasterio():
    try:
        import rasterio
        from rasterio.transform import from_origin
    except ImportError as exc:  # pragma: no cover - depends on optional environment package
        raise FLUSCaseError("writing FLUS GeoTIFF inputs requires rasterio") from exc
    return rasterio, from_origin


def _validate_inputs(
    start_map: np.ndarray,
    probability_cube: np.ndarray,
    class_values: list[int],
    future_demand: dict[int, int],
    restrict_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, dict[int, int], np.ndarray]:
    start = np.asarray(start_map)
    probability = np.asarray(probability_cube, dtype=np.float32)
    classes = [int(cls) for cls in class_values]
    if not classes:
        raise FLUSCaseError("class_values must be non-empty")
    expected_probability_shape = start.shape + (len(classes),)
    if probability.shape != expected_probability_shape:
        raise FLUSCaseError(
            f"probability_cube shape {probability.shape} does not match expected {expected_probability_shape}"
        )
    demand = {int(cls): int(future_demand.get(int(cls), 0)) for cls in classes}
    demand_total = int(sum(demand.values()))
    if demand_total != int(start.size):
        raise FLUSCaseError(f"future demand total {demand_total} does not match raster cells {int(start.size)}")
    if restrict_mask is None:
        restrict = np.ones(start.shape, dtype=np.uint8)
    else:
        restrict = np.asarray(restrict_mask)
        if restrict.shape != start.shape:
            raise FLUSCaseError(f"restrict_mask shape {restrict.shape} does not match start map shape {start.shape}")
        restrict = restrict.astype(np.uint8, copy=False)
    return start.astype(np.int32, copy=False), probability, demand, restrict


def _write_single_band_geotiff(path: Path, array: np.ndarray) -> None:
    rasterio, from_origin = _require_rasterio()
    arr = np.asarray(array)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype=arr.dtype,
        transform=from_origin(0, arr.shape[0], 1, 1),
    ) as dataset:
        dataset.write(arr, 1)


def _write_probability_geotiff(path: Path, probability: np.ndarray) -> None:
    rasterio, from_origin = _require_rasterio()
    cube = np.asarray(probability, dtype=np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=cube.shape[0],
        width=cube.shape[1],
        count=cube.shape[2],
        dtype=cube.dtype,
        transform=from_origin(0, cube.shape[0], 1, 1),
    ) as dataset:
        for band_index in range(cube.shape[2]):
            dataset.write(cube[:, :, band_index], band_index + 1)


def _simulation_config(
    class_values: list[int],
    demand: dict[int, int],
    max_iterations: int,
    neighborhood_size: int,
    accelerated_factor: float,
) -> str:
    n_classes = len(class_values)
    cost_row = ",".join(["1"] * n_classes)
    return "\n".join(
        [
            "[Path of land use data]",
            "landuse.tif",
            "[Path of probability data]",
            "probability.tif",
            "[Path of simulation result]",
            "simresult.tif",
            "[Path of restricted area]",
            "restrict.tif",
            "[Number of types]",
            str(n_classes),
            "[Future Pixels]",
            *[str(int(demand[int(cls)])) for cls in class_values],
            "[Cost Matrix]",
            *[cost_row for _ in class_values],
            "[Intensity of neighborhood]",
            *["1" for _ in class_values],
            "[Maximum Number Of Iterations]",
            str(int(max_iterations)),
            "[Size of neighborhood]",
            str(int(neighborhood_size)),
            "[Accelerated factor]",
            str(float(accelerated_factor)),
            "",
        ]
    )


def _markov_chain_csv(class_values: list[int], demand: dict[int, int], end_year: int) -> str:
    header = ",".join(["year", *[f"type{index + 1}" for index, _ in enumerate(class_values)]])
    row = ",".join([str(int(end_year)), *[str(int(demand[int(cls)])) for cls in class_values]])
    return f"{header}\n{row}\n"


def write_flus_case(
    output_dir: Path,
    start_map: np.ndarray,
    probability_cube: np.ndarray,
    class_values: list[int],
    future_demand: dict[int, int],
    end_year: int,
    restrict_mask: np.ndarray | None = None,
    max_iterations: int = 10,
    neighborhood_size: int = 3,
    accelerated_factor: float = 0.1,
) -> FLUSCasePaths:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    classes = [int(cls) for cls in class_values]
    start, probability, demand, restrict = _validate_inputs(
        start_map=start_map,
        probability_cube=probability_cube,
        class_values=classes,
        future_demand=future_demand,
        restrict_mask=restrict_mask,
    )

    paths = FLUSCasePaths(
        output_dir=output,
        landuse_path=output / "landuse.tif",
        probability_path=output / "probability.tif",
        restrict_path=output / "restrict.tif",
        simulation_config_path=output / "CCregionsimlog.txt",
        markov_chain_path=output / "CCregionMakovChain.csv",
    )
    _write_single_band_geotiff(paths.landuse_path, start)
    _write_probability_geotiff(paths.probability_path, probability)
    _write_single_band_geotiff(paths.restrict_path, restrict)
    paths.simulation_config_path.write_text(
        _simulation_config(
            class_values=classes,
            demand=demand,
            max_iterations=max_iterations,
            neighborhood_size=neighborhood_size,
            accelerated_factor=accelerated_factor,
        ),
        encoding="utf-8",
    )
    paths.markov_chain_path.write_text(_markov_chain_csv(classes, demand, int(end_year)), encoding="utf-8")
    return paths
