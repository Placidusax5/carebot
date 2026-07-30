"""
Hospital Locator Module (Module 6) - PLACEHOLDER data provider.

Without a Google Maps / Places API key configured, this returns a
realistic mock list scoped to whatever city/area string the user supplies,
with working Google Maps search links so users can still find real
directions. Wire up GOOGLE_MAPS_API_KEY in .env + call the Places API
here to replace mock_hospitals() with live data.
"""
import urllib.parse

MOCK_NAMES = [
    "{city} General Hospital",
    "{city} Community Medical Center",
    "St. Mary's Hospital, {city}",
    "{city} Urgent Care Clinic",
    "{city} Family Health Clinic",
]


def _maps_link(query: str) -> str:
    return f"https://www.google.com/maps/search/?api=1&query={urllib.parse.quote(query)}"


def _short_specialty(specialty: str) -> str:
    """Trim a specialty label to a clean word for names/search, e.g.
    'ENT (Otolaryngology)' -> 'ENT', 'Pulmonology / Infectious Disease' -> 'Pulmonology'."""
    return specialty.split("/")[0].split("(")[0].strip()


def get_hospitals(city: str, limit: int = 5) -> list:
    city = city.strip() or "your area"
    hospitals = []
    for i, template in enumerate(MOCK_NAMES[:limit]):
        name = template.format(city=city.title())
        address = f"{100 + i * 15} Main St, {city.title()}"
        hospitals.append({
            "name": name,
            "address": address,
            "phone": f"+1-555-01{i}{i}-{1000 + i * 7}",
            "distance_km": round(1.2 + i * 0.9, 1),
            "maps_link": _maps_link(f"{name}, {address}"),
            "department": None,
        })
    return hospitals


def get_hospitals_for_specialty(
    specialty: str, city: str, limit: int = 4, emergency: bool = False
) -> list:
    """Return hospitals tailored to a medical specialty. The names are
    illustrative, but each Directions link is a real Google Maps search for
    that kind of facility near the given city, so it's genuinely useful."""
    city = (city or "").strip() or "your area"
    city_title = city.title()
    spec = _short_specialty(specialty)

    if emergency:
        search_query = f"emergency hospital near {city}"
        templates = [
            "{city} Emergency Hospital (24/7)",
            "{city} General Hospital — Emergency Dept.",
            "St. Mary's Emergency & Trauma Center, {city}",
        ]
    else:
        search_query = f"{spec} hospital near {city}"
        templates = [
            "{city} {spec} Hospital",
            "{city} {spec} Specialty Center",
            "St. Mary's Hospital — {spec} Dept., {city}",
            "{city} Medical Center ({spec})",
        ]

    hospitals = []
    for i, template in enumerate(templates[:limit]):
        name = template.format(city=city_title, spec=spec)
        address = f"{100 + i * 15} Health Ave, {city_title}"
        hospitals.append({
            "name": name,
            "address": address,
            "phone": f"+1-555-02{i}{i}-{2000 + i * 7}",
            "distance_km": round(1.5 + i * 1.1, 1),
            # Every link is a live Maps search for the right kind of facility.
            "maps_link": _maps_link(search_query),
            "department": specialty,
        })
    return hospitals
