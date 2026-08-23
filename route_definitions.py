"""
WeatherOrNot - location metadata (identity + coordinates only).

No disruption probabilities live here. Those are computed by
`data_pipeline.py` from real fetched data (DGCA national OTP + Open-Meteo
weather, see scripts/fetch_data.py) and cached in
`data_cache/route_probabilities.json`, for every origin-destination pair
within each mode (flight airports x flight airports, train cities x train
cities). The app lets the user pick origin and destination independently.
"""

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

MODES = {"Flight": "flight", "Train": "train"}

# --- Flight locations (major Indian airports) -----------------------------
FLIGHT_LOCATIONS = [
    {"code": "BOM", "name": "Mumbai (BOM)", "coords": (19.0887, 72.8679)},
    {"code": "DEL", "name": "Delhi (DEL)", "coords": (28.5562, 77.1000)},
    {"code": "MAA", "name": "Chennai (MAA)", "coords": (12.9941, 80.1709)},
    {"code": "BLR", "name": "Bangalore (BLR)", "coords": (13.1986, 77.7066)},
    {"code": "CCU", "name": "Kolkata (CCU)", "coords": (22.6547, 88.4467)},
    {"code": "GAU", "name": "Guwahati (GAU)", "coords": (26.1061, 91.5859)},
    {"code": "HYD", "name": "Hyderabad (HYD)", "coords": (17.2403, 78.4294)},
    {"code": "PNQ", "name": "Pune (PNQ)", "coords": (18.5793, 73.9089)},
    {"code": "AMD", "name": "Ahmedabad (AMD)", "coords": (23.0772, 72.6347)},
    {"code": "GOI", "name": "Goa (GOI)", "coords": (15.3800, 73.8310)},
    {"code": "COK", "name": "Kochi (COK)", "coords": (10.1520, 76.4019)},
    {"code": "JAI", "name": "Jaipur (JAI)", "coords": (26.8242, 75.8122)},
]

# --- Train locations (major rail-hub cities) -------------------------------
TRAIN_LOCATIONS = [
    {"code": "BOM-CITY", "name": "Mumbai", "coords": (18.9398, 72.8355)},
    {"code": "NDLS", "name": "New Delhi", "coords": (28.6430, 77.2194)},
    {"code": "MAA-CITY", "name": "Chennai", "coords": (13.0827, 80.2707)},
    {"code": "SBC", "name": "Bangalore", "coords": (12.9716, 77.5946)},
    {"code": "HWH", "name": "Kolkata (Howrah)", "coords": (22.5839, 88.3428)},
    {"code": "TVC", "name": "Trivandrum", "coords": (8.4875, 76.9525)},
    {"code": "MAQ", "name": "Mangalore", "coords": (12.9141, 74.8560)},
    {"code": "PUNE", "name": "Pune", "coords": (18.5293, 73.8743)},
    {"code": "ADI", "name": "Ahmedabad", "coords": (23.0225, 72.5714)},
    {"code": "JP", "name": "Jaipur", "coords": (26.9124, 75.7873)},
    {"code": "LKO", "name": "Lucknow", "coords": (26.8467, 80.9462)},
    {"code": "PNBE", "name": "Patna", "coords": (25.5941, 85.1376)},
]

# Fallback illustrative (ticket price, insurance cost), since there's no
# longer a single curated route to hang defaults off of - user edits these
# freely. Insurance cost is roughly ~3-4% of ticket price, in line with the
# report's own IRDAI-cited travel insurance pricing benchmarks.
DEFAULT_TICKET_INSURANCE = {
    "flight": (5000, 200),
    "train": (1200, 35),
}


def get_locations(mode):
    return FLIGHT_LOCATIONS if mode == "Flight" else TRAIN_LOCATIONS


def location_name(mode_key, code):
    locations = FLIGHT_LOCATIONS if mode_key == "flight" else TRAIN_LOCATIONS
    for loc in locations:
        if loc["code"] == code:
            return loc["name"]
    return code


def route_id(mode_key, origin_code, dest_code):
    return f"{mode_key}:{origin_code}-{dest_code}"


def unique_locations():
    """Dedup (code -> (lat, lon)) across every flight/train location, for the weather fetcher."""
    locations = {}
    for loc in FLIGHT_LOCATIONS + TRAIN_LOCATIONS:
        locations[loc["code"]] = loc["coords"]
    return locations
