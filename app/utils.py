def round_floats(
    obj: float | dict | list, n_decimal_places: int = 4
) -> float | dict | list:
    if isinstance(obj, float):
        return round(obj, n_decimal_places)
    if isinstance(obj, dict):
        return {k: round_floats(v, n_decimal_places) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_floats(x, n_decimal_places) for x in obj]
    return obj
