import numpy as np

from ornaments.wallpaper_groups import get_subgroups, group_labels, group_name, calculate_orbit, grid_points, \
    replicate_tiles, extract_tiles, create_wallpaper_pattern, extract_fundamental_region


def test_get_subgroups():
    assert set(get_subgroups('P6m')) == {'P6', 'P31m', 'P3m1', 'Cmm', 'P3', 'Cm', 'P2', 'P1'}
    assert set(get_subgroups('P4g')) == {'P4', 'Pgg', 'P2', 'P1'}


def test_group_name():
    for group, label in group_labels.items():
        assert group_name(label) == group


def test_calculate_orbit():
    p = (1, 1)
    orbit1 = {(1, 1), (1, 2), (2, 1), (2, 2)}
    orbit2 = calculate_orbit([p], 'P1', (0, 0), (1, 0), (0, 1), 1, 2, 1, 2)
    orbit2 = set(tuple(x[0]) for x in orbit2)

    assert orbit1 == orbit2


def test_grid_points():
    grid1 = {(0, 0), (1, 0), (0, 1), (1, 1)}
    grid2 = set(tuple(x) for x in grid_points((0, 0), (1, 0), (0, 1), 2, 2))

    assert grid1 == grid2


def test_create_wallpaper_pattern():
    t0, t1, t2 = (16, 16), (64, 0), (0, 64)
    indices = np.indices((64, 32)).reshape(2, -1).T
    values = np.random.rand(64 * 32, 3)
    tile = np.empty((64, 64, 3))
    tile[:32, :] = values.reshape((64, 32, 3)).transpose(1, 0, 2)
    tile[32:, :] = values.reshape((64, 32, 3)).transpose(1, 0, 2)[::-1, ::-1]
    pattern = np.tile(tile, (5, 5, 1))[48:-16, 48:-16]

    assert np.allclose(create_wallpaper_pattern(indices, values, 'P2', t0, t1, t2, 256, 256), pattern)


def test_extract_fundamental_region():
    t0, t1, t2 = (32, 16), (64, 0), (0, 64)
    indices = np.indices((64, 32)).reshape(2, -1).T
    values = np.random.rand(64 * 32, 3)
    pattern = create_wallpaper_pattern(indices, values, 'P2', t0, t1, t2, 256, 256)

    indices2, values2 = extract_fundamental_region(pattern, 'P2', t0, t1, t2)

    assert indices2.shape == indices.shape
    assert np.all(indices2.max(0) == indices.max(0))
    assert np.allclose(values2.reshape((64, 32, 3)), values.reshape((64, 32, 3)))


def test_replicate_tiles():
    t0, t1, t2 = (32, 16), (64, 0), (0, 64)
    indices = np.indices((64, 32)).reshape(2, -1).T
    values = np.random.rand(64 * 32, 3)
    pattern = create_wallpaper_pattern(indices, values, 'P2', t0, t1, t2, 256, 256)

    assert np.allclose(replicate_tiles([pattern], [t0], [t1], [t2]), pattern)


def test_extract_tiles():
    t0, t1, t2 = (32, 16), (64, 0), (0, 64)
    indices = np.indices((64, 32)).reshape(2, -1).T
    values = np.random.rand(64 * 32, 3)
    tile = np.empty((64, 64, 3))
    tile[:32, :] = values.reshape((64, 32, 3)).transpose(1, 0, 2)
    tile[32:, :] = values.reshape((64, 32, 3)).transpose(1, 0, 2)[::-1, ::-1]
    pattern = create_wallpaper_pattern(indices, values, 'P2', t0, t1, t2, 256, 256)

    assert np.allclose(extract_tiles([pattern], [t0], [t1], [t2], 64, 64), [tile])
