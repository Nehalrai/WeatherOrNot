"""
WeatherOrNot - real, primary-source evidence that weather is a documented,
quantified cause of flight disruption in India.

Source: Government of India, Ministry of Civil Aviation, Rajya Sabha
Unstarred Question No. 192 (answered 5 February 2024, Minister of State
Gen. (Dr) V. K. Singh), "Delayed Flights Due to Extreme Weather":
    https://sansad.in/getFile/annex/263/AU192.pdf?source=pqars
(surfaced via https://dataful.in/datasets/19650/, "Year and Airline-wise
Scheduled Flights Delayed due to various reasons")

The PDF is cached at data_cache/AU192_rajya_sabha_weather_delays.pdf by
scripts/fetch_data.py for auditability. Its two annexure tables are
transcribed below (annual, 2021-2023, major scheduled carriers only - this
is genuinely coarser than the monthly/route-level data used elsewhere in
this app, which is why it's used as supplementary cited evidence rather
than as an input to the EV probability calculation itself - see the note
in data_pipeline.py on why the raw "Weather" reason code understates true
weather-linked disruption (most of it shows up downstream as
"Reactionary" knock-on delay, not "Weather").
"""

SOURCE_CITATION = (
    "Rajya Sabha Unstarred Question No. 192, Ministry of Civil Aviation, "
    "answered 5 February 2024 (sansad.in/getFile/annex/263/AU192.pdf)"
)

# Annexure-II: Total departures + delay-reason breakdown, major scheduled
# carriers, by year. Reason order: Passenger&Baggage, Ramp, Technical,
# Operational, Reactionary, Airport, Weather, ATC, Misc.
REASON_ORDER = [
    "passenger_baggage", "ramp", "technical", "operational",
    "reactionary", "airport", "weather", "atc", "misc",
]

DELAY_REASONS_BY_YEAR = {
    2023: [
        ("Indigo", 236298, [1059, 559, 476, 1999, 23553, 661, 1595, 4403, 191]),
        ("Alliance Air", 11271, [84, 66, 351, 316, 2782, 48, 201, 356, 10]),
        ("Air India", 58291, [509, 368, 896, 2051, 8440, 778, 350, 1481, 73]),
        ("Go First", 10678, [66, 35, 283, 696, 4035, 22, 86, 201, 4]),
        ("Air Asia", 33876, [199, 56, 274, 195, 6026, 190, 217, 435, 48]),
        ("Akasa Air", 23192, [25, 10, 114, 176, 2521, 76, 298, 553, 13]),
        ("Spicejet", 25003, [260, 184, 789, 616, 7253, 160, 269, 830, 34]),
        ("Vistara", 58733, [223, 40, 86, 796, 7989, 347, 414, 1437, 38]),
    ],
    2022: [
        ("Indigo", 196295, [1009, 272, 396, 1699, 16297, 636, 1110, 2177, 272]),
        ("Alliance Air", 11498, [131, 35, 356, 186, 2314, 57, 139, 75, 28]),
        ("Air India", 44807, [508, 265, 547, 468, 3361, 321, 143, 405, 30]),
        ("Go First", 35790, [142, 63, 803, 1157, 6296, 77, 128, 469, 13]),
        ("Air Asia", 27021, [127, 8, 111, 111, 2883, 71, 115, 68, 85]),
        ("Spicejet", 33825, [156, 133, 968, 292, 5007, 149, 259, 568, 47]),
        ("Vistara", 53180, [180, 15, 105, 424, 3915, 341, 212, 736, 31]),
    ],
    2021: [
        ("Indigo", 153367, [805, 215, 299, 637, 6919, 495, 895, 1194, 150]),
        ("Alliance Air", 1653, [29, 3, 31, 68, 373, 14, 37, 20, 1]),
        ("Air India", 42243, [322, 200, 475, 552, 4583, 350, 300, 332, 83]),
        ("Go First", 24439, [41, 23, 80, 201, 2969, 72, 96, 473, 19]),
        ("Air Asia", 21236, [54, 8, 105, 77, 2085, 66, 114, 84, 14]),
        ("Spicejet", 31654, [108, 53, 506, 177, 3278, 189, 351, 321, 21]),
        ("Vistara", 33806, [80, 12, 67, 195, 2636, 227, 191, 205, 38]),
    ],
}


def summarize():
    """Real, computed (not hand-typed) national weather-delay statistics
    per year, from the transcribed annexure tables above."""
    weather_idx = REASON_ORDER.index("weather")
    reactionary_idx = REASON_ORDER.index("reactionary")

    by_year = {}
    for year, rows in DELAY_REASONS_BY_YEAR.items():
        total_departures = sum(r[1] for r in rows)
        weather = sum(r[2][weather_idx] for r in rows)
        reactionary = sum(r[2][reactionary_idx] for r in rows)
        all_reason_coded = sum(sum(r[2]) for r in rows)
        by_year[year] = {
            "total_departures": total_departures,
            "weather_delays": weather,
            "reactionary_delays": reactionary,
            "all_reason_coded_delays": all_reason_coded,
            "weather_delay_rate": round(weather / total_departures, 4),
            "weather_share_of_reason_coded_delays": round(weather / all_reason_coded, 4),
        }

    years = sorted(by_year)
    avg_weather_share = sum(by_year[y]["weather_share_of_reason_coded_delays"] for y in years) / len(years)
    avg_weather_delay_rate = sum(by_year[y]["weather_delay_rate"] for y in years) / len(years)

    return {
        "source": SOURCE_CITATION,
        "years_covered": years,
        "by_year": by_year,
        "avg_weather_share_of_reason_coded_delays": round(avg_weather_share, 4),
        "avg_weather_delay_rate_vs_total_departures": round(avg_weather_delay_rate, 4),
        "note": (
            "The raw 'Weather' reason code alone understates true weather-linked "
            "disruption: 'Reactionary' (knock-on delay from an earlier flight) is "
            "consistently the largest category (~70% of reason-coded delays) and "
            "is frequently a downstream effect of an earlier weather-caused delay "
            "that gets bureaucratically recoded once it cascades. This evidence is "
            "used to substantiate that weather is a real, DGCA-documented cause of "
            "disruption - it is not used as a direct multiplier in the EV "
            "probability calculation, which instead uses the fuller all-cause "
            "monthly DGCA OTP baseline (see data_pipeline.py) precisely because "
            "that one does capture reactionary/cascading effects."
        ),
    }
