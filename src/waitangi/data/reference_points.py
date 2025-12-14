"""Reference points for validating Waitangi estuary geometry.

Ground-truth points used to validate water/land classification:
- WATER_POINTS: Known navigable water (should always be classified as water)
- LAND_POINTS: Known land areas near water
- NEVER_UNDERWATER_POINTS: Points that must NEVER be flooded at any tide level
- LANDMARKS: Notable locations for orientation

These are known ground-truth coordinates provided by local knowledge
to verify that geometry, flooding models, and rendering are correct.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ReferencePoint:
    """A known reference point with lat/lon and description."""
    lat: float
    lon: float
    name: str
    point_type: str  # "water", "land", "landmark"


# Key landmarks with exact coordinates
LANDMARKS = [
    ReferencePoint(-35.270798, 174.078968, "Boat Ramp", "landmark"),
    ReferencePoint(-35.272709, 174.079625, "Waitangi Bridge (Paihia side)", "landmark"),
    ReferencePoint(-35.271170, 174.079652, "Waitangi Bridge (Waitangi side)", "landmark"),
    ReferencePoint(-35.278284, 174.051297, "Haruru Falls", "landmark"),
]

# Points that are KNOWN TO BE IN WATER (for validation)
# West of the bridge (estuary/river going upstream)
WATER_POINTS_WEST = [
    ReferencePoint(-35.273205, 174.077059, "West of bridge 1", "water"),
    ReferencePoint(-35.268631, 174.071339, "West of bridge 2", "water"),
    ReferencePoint(-35.273630, 174.071454, "West of bridge 3", "water"),
    ReferencePoint(-35.281389, 174.067742, "West of bridge 4", "water"),
    ReferencePoint(-35.276570, 174.066071, "West of bridge 5", "water"),
    ReferencePoint(-35.273510, 174.059983, "West of bridge 6", "water"),
    ReferencePoint(-35.274449, 174.055009, "West of bridge 7", "water"),
    ReferencePoint(-35.275995, 174.053969, "West of bridge 8", "water"),
    ReferencePoint(-35.278571, 174.053783, "Near Haruru Falls", "water"),
]

# East of the bridge (bay/ocean side)
WATER_POINTS_EAST = [
    ReferencePoint(-35.273206, 174.081181, "East of bridge 1", "water"),
    ReferencePoint(-35.268721, 174.088085, "East of bridge 2", "water"),
    ReferencePoint(-35.262114, 174.096178, "East of bridge 3 (bay)", "water"),
    ReferencePoint(-35.279267, 174.102713, "East of bridge 4 (bay)", "water"),
    ReferencePoint(-35.273180, 174.117051, "East of bridge 5 (outer bay)", "water"),
    ReferencePoint(-35.238829, 174.099914, "East of bridge 6 (outer bay)", "water"),
]

# Points that are KNOWN TO BE ON LAND (for validation)
# These should never be flooded at any tide level
# Includes roads, buildings, and other permanent land features
LAND_POINTS = [
    # Original land points
    ReferencePoint(-35.275846, 174.071482, "Land 1 (central)", "land"),
    ReferencePoint(-35.278834, 174.071211, "Land 2 (south)", "land"),
    ReferencePoint(-35.275274, 174.057311, "Land 3 (west)", "land"),
    ReferencePoint(-35.272980, 174.054596, "Land 4 (west)", "land"),
    ReferencePoint(-35.277560, 174.053186, "Land 5 (Haruru)", "land"),
    ReferencePoint(-35.277275, 174.055392, "Land 6 (Haruru)", "land"),
    ReferencePoint(-35.279056, 174.051033, "Land 7 (Haruru)", "land"),
    ReferencePoint(-35.277416, 174.052795, "Land 8 (Haruru)", "land"),
    # Near the bridge area
    ReferencePoint(-35.276600, 174.074886, "Land 9 (near bridge)", "land"),
    ReferencePoint(-35.274998, 174.078428, "Land 10 (near bridge)", "land"),
    # Western estuary area - roads/land near mangroves
    ReferencePoint(-35.275725, 174.061614, "Land 11 (west)", "land"),
    ReferencePoint(-35.274107, 174.057801, "Land 12 (west)", "land"),
    ReferencePoint(-35.275878, 174.062021, "Land 13 (west)", "land"),
    ReferencePoint(-35.276902, 174.064400, "Land 14 (west)", "land"),
    # Near Haruru Falls
    ReferencePoint(-35.278996, 174.054232, "Land 15 (Haruru)", "land"),
    ReferencePoint(-35.278396, 174.055046, "Land 16 (Haruru)", "land"),
    ReferencePoint(-35.277934, 174.053175, "Land 17 (Haruru)", "land"),
    ReferencePoint(-35.278556, 174.051303, "Land 18 (Haruru)", "land"),
    ReferencePoint(-35.279024, 174.053451, "Land 19 (Haruru)", "land"),
    # Eastern/bay side
    ReferencePoint(-35.270745, 174.078994, "Land 20 (east)", "land"),
    ReferencePoint(-35.268544, 174.076922, "Land 21 (east)", "land"),
]

# NEVER_UNDERWATER_POINTS is now merged into LAND_POINTS above
# Kept as empty list for backward compatibility with any code that imports it
NEVER_UNDERWATER_POINTS: list[ReferencePoint] = []

# Mangrove areas - water flows THROUGH them but not OVER them
# These are intertidal zones where:
# - The mudflat floods at high tide
# - But the mangrove vegetation stays above water
# - Water cannot flow freely (blocked by roots/vegetation)
# - NOT navigable by kayak
MANGROVE_POINTS = [
    # Near the bridge / eastern estuary
    ReferencePoint(-35.275364, 174.075757, "Mangrove 1 (near bridge)", "mangrove"),
    ReferencePoint(-35.275773, 174.076320, "Mangrove 2 (near bridge)", "mangrove"),
    ReferencePoint(-35.275453, 174.077326, "Mangrove 3 (near bridge)", "mangrove"),
    ReferencePoint(-35.270664, 174.073043, "Mangrove 4 (north)", "mangrove"),
    # Central estuary
    ReferencePoint(-35.269906, 174.067780, "Mangrove 5 (central)", "mangrove"),
    ReferencePoint(-35.266705, 174.071456, "Mangrove 6 (north central)", "mangrove"),
    ReferencePoint(-35.267638, 174.065316, "Mangrove 7 (north)", "mangrove"),
    ReferencePoint(-35.267905, 174.066922, "Mangrove 8 (north)", "mangrove"),
    ReferencePoint(-35.272490, 174.066707, "Mangrove 9 (central)", "mangrove"),
    ReferencePoint(-35.274334, 174.068230, "Mangrove 10 (central)", "mangrove"),
    ReferencePoint(-35.273405, 174.068870, "Mangrove 11 (central)", "mangrove"),
    # Southern estuary
    ReferencePoint(-35.278169, 174.067832, "Mangrove 12 (south)", "mangrove"),
    ReferencePoint(-35.279262, 174.065898, "Mangrove 13 (south)", "mangrove"),
    ReferencePoint(-35.278182, 174.064509, "Mangrove 14 (south)", "mangrove"),
    ReferencePoint(-35.281538, 174.065926, "Mangrove 15 (south)", "mangrove"),
    ReferencePoint(-35.281138, 174.069302, "Mangrove 16 (south)", "mangrove"),
    ReferencePoint(-35.283416, 174.066436, "Mangrove 17 (far south)", "mangrove"),
    ReferencePoint(-35.284479, 174.063739, "Mangrove 18 (far south)", "mangrove"),
    ReferencePoint(-35.284971, 174.064380, "Mangrove 19 (far south)", "mangrove"),
    # Western estuary (toward Haruru)
    ReferencePoint(-35.272342, 174.059430, "Mangrove 20 (west)", "mangrove"),
    ReferencePoint(-35.279227, 174.063914, "Mangrove 21 (southwest)", "mangrove"),
    ReferencePoint(-35.267283, 174.071309, "Mangrove 22 (north)", "mangrove"),
    # Reclassified from land points (flooded at high tide)
    ReferencePoint(-35.270782, 174.072122, "Mangrove 23 (north, ex-land 1)", "mangrove"),
    ReferencePoint(-35.271282, 174.067321, "Mangrove 24 (north, ex-land 2)", "mangrove"),
    ReferencePoint(-35.273653, 174.068428, "Mangrove 25 (central, ex-land 3)", "mangrove"),
    ReferencePoint(-35.273218, 174.063054, "Mangrove 26 (west, ex-land 4)", "mangrove"),
    ReferencePoint(-35.280260, 174.066867, "Mangrove 27 (south, ex-land 6)", "mangrove"),
]

# All water points combined
ALL_WATER_POINTS = WATER_POINTS_WEST + WATER_POINTS_EAST

# All reference points
ALL_POINTS = LANDMARKS + ALL_WATER_POINTS + LAND_POINTS + NEVER_UNDERWATER_POINTS + MANGROVE_POINTS


def get_bounding_box() -> tuple[float, float, float, float]:
    """Get bounding box that contains all reference points.

    Returns:
        (min_lat, min_lon, max_lat, max_lon)
    """
    lats = [p.lat for p in ALL_POINTS]
    lons = [p.lon for p in ALL_POINTS]
    margin = 0.005  # ~500m margin
    return (
        min(lats) - margin,
        min(lons) - margin,
        max(lats) + margin,
        max(lons) + margin,
    )


def validate_point_classification(
    classify_func: callable,
) -> dict[str, list[tuple[ReferencePoint, str]]]:
    """Validate a classification function against known reference points.

    Args:
        classify_func: Function that takes (lat, lon) and returns "water" or "land"

    Returns:
        Dictionary with "correct", "incorrect", and "unknown" lists
    """
    results = {"correct": [], "incorrect": [], "unknown": []}

    for point in ALL_WATER_POINTS + LAND_POINTS:
        try:
            classification = classify_func(point.lat, point.lon)
            if classification == point.point_type:
                results["correct"].append((point, classification))
            else:
                results["incorrect"].append((point, classification))
        except Exception as e:
            results["unknown"].append((point, str(e)))

    return results
