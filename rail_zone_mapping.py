"""
WeatherOrNot - maps the 12 train cities in route_definitions.py to Indian
Railway zones, and those zones to the zone strings used by the delay dataset.

Used by data_pipeline.py to look up zone-level disruption rates instead of
scaling a single national punctuality figure by weather.
"""

import re

# Station code (route_definitions.TRAIN_LOCATIONS) -> zone full name.
#
# ponytail: assignments are as specified by the project brief, not re-derived
# from Indian Railways' official zone/division boundaries. Two differ from the
# official allocation - see ZONE_ASSIGNMENT_CAVEATS below.
CITY_ZONE = {
    "BOM-CITY": "Western Railway",        # Mumbai
    "NDLS":     "Northern Railway",       # New Delhi
    "MAA-CITY": "Southern Railway",       # Chennai
    "SBC":      "South Western Railway",  # Bangalore
    "HWH":      "Eastern Railway",        # Kolkata (Howrah)
    "TVC":      "Southern Railway",       # Trivandrum
    "MAQ":      "South Western Railway",  # Mangalore
    "PUNE":     "Central Railway",        # Pune
    "ADI":      "Western Railway",        # Ahmedabad
    "JP":       "North Western Railway",  # Jaipur
    "LKO":      "North Central Railway",  # Lucknow
    "PNBE":     "East Central Railway",   # Patna
}

# Zone full name -> the code the dataset's station table uses for that zone.
ZONE_DATASET_CODE = {
    "Central Railway": "CR",
    "Eastern Railway": "ER",
    "East Central Railway": "ECR",
    "East Coast Railway": "ECoR",
    "Northern Railway": "NR",
    "North Central Railway": "NCR",
    "North Eastern Railway": "NER",
    "Northeast Frontier Railway": "NFR",
    "North Western Railway": "NWR",
    "Southern Railway": "SR",
    "South Central Railway": "SCR",
    "South Eastern Railway": "SER",
    "South East Central Railway": "SECR",
    "South Western Railway": "SWR",
    "Western Railway": "WR",
    "West Central Railway": "WCR",
    "Konkan Railway": "KR",
}

# Flagged for the record: the brief's assignment differs from Indian Railways'
# official allocation for these two. Kept as specified; change the dict above
# if you want the official zones instead.
ZONE_ASSIGNMENT_CAVEATS = {
    "MAQ": "Mangalore is officially Southern Railway (Palakkad div) / Konkan Railway, not South Western.",
    "LKO": "Lucknow Charbagh (LKO) is officially Northern Railway; Lucknow Jn (LJN) is North Eastern. Neither is North Central.",
}


def zone_for(station_code):
    """Zone full name for a TRAIN_LOCATIONS station code."""
    return CITY_ZONE[station_code]


def zone_code_for(station_code):
    """Dataset zone code (e.g. 'WR') for a TRAIN_LOCATIONS station code."""
    return ZONE_DATASET_CODE[CITY_ZONE[station_code]]


def normalize_zone(raw):
    """Map a zone string from the dataset ('WR', 'Western Railway', 'W.Rly') to
    its full name, or None if unrecognised."""
    if not raw:
        return None
    cleaned = raw.strip()
    if cleaned in ZONE_DATASET_CODE:
        return cleaned
    # Compact to letters only; a trailing "RLY"/"RAILWAY" means the same as the
    # "R" that ends every zone code, so fold it down to that.
    compact = re.sub(r"[^A-Za-z]", "", cleaned).upper()
    compact = re.sub(r"(RAILWAY|RLY)$", "R", compact)
    for full, code in ZONE_DATASET_CODE.items():
        if compact == code.upper() or compact == re.sub(r"[^A-Za-z]", "", full).upper():
            return full
    return None


def demo():
    from route_definitions import TRAIN_LOCATIONS
    codes = {loc["code"] for loc in TRAIN_LOCATIONS}
    assert codes == set(CITY_ZONE), f"drift vs route_definitions: {codes ^ set(CITY_ZONE)}"
    assert all(z in ZONE_DATASET_CODE for z in CITY_ZONE.values())
    assert zone_code_for("BOM-CITY") == "WR"
    assert zone_code_for("PNBE") == "ECR"
    assert normalize_zone("WR") == "Western Railway"
    assert normalize_zone("Western Railway") == "Western Railway"
    assert normalize_zone("S.W.Rly") == "South Western Railway"
    assert normalize_zone("") is None and normalize_zone("XYZ") is None
    print(f"ok - {len(CITY_ZONE)} cities across {len(set(CITY_ZONE.values()))} zones")


if __name__ == "__main__":
    demo()
