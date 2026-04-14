"""Shared constants for the value_probe suite.

No model loading here — just data definitions used by all three scripts.
"""

# ---------------------------------------------------------------------------
# Layer to analyse (BERT L4 = index 4 in 1-indexed, i.e. hidden_states[4])
# ---------------------------------------------------------------------------
VALUE_LAYER = 4  # 1-indexed; hidden_states[4] is the 4th transformer block

# ---------------------------------------------------------------------------
# Denominations: (name, pence_value)  — full 12-coin axis, farthing to guinea
# Axis direction is defined by farthing (low) and guinea (high) — both
# unambiguous coin names.  Sovereign and crown are included for evaluation
# (to see where BERT places them) but do not define the axis.
# ---------------------------------------------------------------------------
DENOMINATIONS = [
    ("farthing",       0.25),
    ("halfpenny",      0.5),
    ("penny",          1.0),
    ("threepence",     3.0),
    ("groat",          4.0),
    ("sixpence",       6.0),
    ("shilling",       12.0),
    ("florin",         24.0),
    ("half-crown",     30.0),
    ("crown",          60.0),
    ("half-sovereign", 120.0),
    ("sovereign",      240.0),
    ("guinea",         252.0),
]

# ---------------------------------------------------------------------------
# Templates for embedding a denomination in a neutral (no-year) context
# ---------------------------------------------------------------------------
VALUE_TEMPLATES = [
    "the coin was worth {c} .",
    "i paid {c} for it .",
    "the price was one {c} .",
    "a {c} changed hands .",
]

# ---------------------------------------------------------------------------
# Templates for year-probe sentences  (explicit year carrier)
# ---------------------------------------------------------------------------
YEAR_TEMPLATES = [
    "the year is {y} .",
    "it is the year {y} .",
    "this took place in the year {y} .",
    "the date is {y} .",
]

# ---------------------------------------------------------------------------
# Year range for building the year-probe time axis
# ---------------------------------------------------------------------------
YEAR_PROBE_YEARS = list(range(1200, 1901, 25))   # 1200, 1225, … 1900

# ---------------------------------------------------------------------------
# Test years for the coin × time 2D probe
# ---------------------------------------------------------------------------
TEST_YEARS = list(range(1250, 1921, 10))  # 1250, 1260, … 1920  (68 points)

# ---------------------------------------------------------------------------
# Test coins for Part B of the main probe (subset of DENOMINATIONS)
# ---------------------------------------------------------------------------
TEST_COINS = [
    ("penny",    1),
    ("sixpence", 6),
    ("shilling", 12),
    ("florin",   24),
]

# ---------------------------------------------------------------------------
# Templates for embedding a coin in an explicit year context
# ---------------------------------------------------------------------------
TIME_TEMPLATES = [
    "in {y} , a {c} was used for everyday purchases .",
    "the {c} was a common coin in {y} .",
    "in {y} , people paid with a {c} .",
    "a {c} changed hands in {y} .",
]

# ---------------------------------------------------------------------------
# Sequence definitions  (4 monotonic sequences from sequence_geometry_probe)
# Each entry: dict with keys  title, periods, templates
# periods: list of (label, start_year, end_year)
# ---------------------------------------------------------------------------

SEQUENCES = {
    "ruling_dynasty": {
        "title": "Ruling dynasty",
        "monotonic": True,
        "periods": [
            ("Plantagenet", 1200, 1398),
            ("Lancaster",   1399, 1460),
            ("York",        1461, 1484),
            ("Tudor",       1485, 1602),
            ("Stuart",      1603, 1713),
            ("Hanover",     1714, 1836),
            ("Windsor",     1837, 1936),
        ],
        "templates": [
            "the {label} dynasty ruled england .",
            "the house of {label} held the throne .",
            "england was ruled by the {label} family .",
            "the {label} monarch sat on the english throne .",
        ],
    },

    "primary_weapon": {
        "title": "Primary weapon",
        "monotonic": True,
        "periods": [
            ("longbow", 1200, 1499),
            ("pike",    1500, 1649),
            ("musket",  1650, 1849),
            ("rifle",   1850, 1930),
        ],
        "templates": [
            "the soldier carried a {label} into battle .",
            "troops were armed with the {label} .",
            "the primary weapon of the infantry was the {label} .",
            "soldiers fought with the {label} .",
        ],
    },

    "ship_construction": {
        "title": "Ship construction",
        "monotonic": True,
        "periods": [
            ("wooden ship", 1200, 1819),
            ("iron ship",   1820, 1869),
            ("steel ship",  1870, 1930),
        ],
        "templates": [
            "the navy sailed in a {label} .",
            "the vessel was a {label} .",
            "the fleet consisted of {label}s .",
            "the warship was a {label} .",
        ],
    },

    "primary_fuel": {
        "title": "Primary fuel",
        "monotonic": True,
        "periods": [
            ("wood and peat", 1200, 1699),
            ("coal",          1700, 1819),
            ("steam coal",    1820, 1930),
        ],
        "templates": [
            "homes were heated with {label} .",
            "the furnace burned {label} .",
            "industry relied on {label} for energy .",
            "heat was produced by burning {label} .",
        ],
    },
}
