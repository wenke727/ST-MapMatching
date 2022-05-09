from math import radians, cos, sin, asin, sqrt
from enum import Enum
import numpy as np


# mean earth radius - https://en.wikipedia.org/wiki/Earth_radius#Mean_radius
_AVG_EARTH_RADIUS_KM = 6371.0088


class Unit(Enum):
    """
    Enumeration of supported units.
    The full list can be checked by iterating over the class; e.g.
    the expression `tuple(Unit)`.
    """

    KILOMETERS = 'km'
    METERS = 'm'
    MILES = 'mi'
    NAUTICAL_MILES = 'nmi'
    FEET = 'ft'
    INCHES = 'in'


# Unit values taken from http://www.unitconversion.org/unit_converter/length.html
_CONVERSIONS = {Unit.KILOMETERS:       1.0,
                Unit.METERS:           1000.0,
                Unit.MILES:            0.621371192,
                Unit.NAUTICAL_MILES:   0.539956803,
                Unit.FEET:             3280.839895013,
                Unit.INCHES:           39370.078740158}


def haversine(point1, point2, unit=Unit.KILOMETERS):
    """ Calculate the great-circle distance between two points on the Earth surface.

    Takes two 2-tuples, containing the latitude and longitude of each point in decimal degrees,
    and, optionally, a unit of length.

    :param point1: first point; tuple of (latitude, longitude)(纬度, 经度) in decimal degrees
    :param point2: second point; tuple of (latitude, longitude) in decimal degrees
    :param unit: a member of haversine.Unit, or, equivalently, a string containing the
                 initials of its corresponding unit of measurement (i.e. miles = mi)
                 default 'km' (kilometers).

    Example: ``haversine((45.7597, 4.8422), (48.8567, 2.3508), unit=Unit.METERS)``

    Precondition: ``unit`` is a supported unit (supported units are listed in the `Unit` enum)

    :return: the distance between the two points in the requested unit, as a float.

    The default returned unit is kilometers. The default unit can be changed by
    setting the unit parameter to a member of ``haversine.Unit``
    (e.g. ``haversine.Unit.INCHES``), or, equivalently, to a string containing the
    corresponding abbreviation (e.g. 'in'). All available units can be found in the ``Unit`` enum.
    """

    # get earth radius in required units
    unit = Unit(unit)
    avg_earth_radius = _AVG_EARTH_RADIUS_KM * _CONVERSIONS[unit]

    # unpack latitude/longitude
    lat1, lng1 = point1
    lat2, lng2 = point2

    # convert all latitudes/longitudes from decimal degrees to radians
    lat1, lng1, lat2, lng2 = map(radians, (lat1, lng1, lat2, lng2))

    # calculate haversine
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = sin(lat * 0.5) ** 2 + cos(lat1) * cos(lat2) * sin(lng * 0.5) ** 2

    return 2 * avg_earth_radius * asin(sqrt(d))


def haversine_np(point1, point2, xy=True, unit=Unit.KILOMETERS):
    """ Calculate the great-circle distance between two points array on the Earth surface.

    Takes two 2-tuples, containing the latitude and longitude of each point in decimal degrees,
    and, optionally, a unit of length.

    :param point1: first serial points; tuple of (latitude, longitude)(纬度, 经度) in decimal degrees in default
    :param point2: second serial points; tuple of (latitude, longitude) in decimal degrees in default
    :param xy: if True, (latitude, longitude)(纬度, 经度) ; other wise (longitude, latitude)
    :param unit: a member of haversine.Unit, or, equivalently, a string containing the
                 initials of its corresponding unit of measurement (i.e. miles = mi)
                 default 'km' (kilometers).

    Example: ``haversine((45.7597, 4.8422), (48.8567, 2.3508), unit=Unit.METERS)``

    Precondition: ``unit`` is a supported unit (supported units are listed in the `Unit` enum)

    :return: the distance between the two points in the requested unit, as a float.

    The default returned unit is kilometers. The default unit can be changed by
    setting the unit parameter to a member of ``haversine.Unit``
    (e.g. ``haversine.Unit.INCHES``), or, equivalently, to a string containing the
    corresponding abbreviation (e.g. 'in'). All available units can be found in the ``Unit`` enum.
    """

    # get earth radius in required units
    unit = Unit(unit)
    avg_earth_radius = _AVG_EARTH_RADIUS_KM * _CONVERSIONS[unit]

    # unpack latitude/longitude
    if xy:
        lat1, lng1 = point1[:, 1], point1[:, 0]
        lat2, lng2 = point2[:, 1], point2[:, 0]    
    else:
        lat1, lng1 = point1[:, 0], point1[:, 1]
        lat2, lng2 = point2[:, 0], point2[:, 1]

    # convert all latitudes/longitudes from decimal degrees to radians
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))

    # calculate haversine
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2

    return 2 * avg_earth_radius * np.arcsin(np.sqrt(d))



if __name__ == "__main__":
    from shapely import wkt
    
    polyline = wkt.loads("LINESTRING (113.934186 22.57795, 113.934227 22.577982, 113.934274 22.578013, 113.934321 22.578035, 113.934373 22.578052, 113.934421 22.57806, 113.93448 22.578067)")
    coords = np.array(polyline.coords[:])
    sum(haversine_np(coords[:-1], coords[1:], unit=Unit.METERS))

    in_crs:int=4326
    out_crs:int=900913
    import geopandas as gpd
    tmp = gpd.GeoSeries([polyline]).set_crs(in_crs, allow_override=True).to_crs(out_crs)
    l_ = tmp.loc[0]
    l_.length
    
    