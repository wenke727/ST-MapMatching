""" Global config """
from pathlib import Path

IP = "192.168.135.16"
postgre_url= f"postgresql://postgres:pcl_A5A@{IP}:5432/gis"

root = Path(__file__).parent
DEBUG_FOLDER = root / "../debug"
LOG_FOLDER   = root / "../log"
DATA_FOLDER  = root / "../data"

DIS_FACTOR = 1/110/1000

GBA_BBOX = [111.35669933,  21.56670092, 115.41989933,  24.39190092]
SZ_BBOX  = [113.746280,  22.441466, 114.623972,  22.864722]
PCL_BBOX = [113.930914,  22.570536, 113.945456,  22.585613]
FT_BBOX  = [114.05097,   22.53447,  114.05863,   22.54605]


""" road_type_filter """
# Note: we adopt the filter logic from osmnx (https://github.com/gboeing/osmnx)
# exclude links with tag attributes in the filters
filters = {}


# 道路含义：'service'：通往设施的道路
filters['auto'] = {'area':['yes'],
                   'highway':['cycleway','footway','path','pedestrian','steps','track','corridor','elevator','escalator',
                              'proposed','construction','bridleway','abandoned','platform','raceway'],
                   'motor_vehicle':['no'],
                   'motorcar':['no'],
                   'access':['private'],
                   'service':['parking','parking_aisle','driveway','private','emergency_access']
                   }

filters['bike'] = {'area':['yes'],
                   'highway':['footway','steps','corridor','elevator','escalator','motor','proposed','construction','abandoned','platform','raceway'],
                   'bicycle':['no'],
                   'service':['private'],
                   'access':['private']
                   }

filters['walk'] = {'area':['yes'],
                   'highway':['cycleway','motor','proposed','construction','abandoned','platform','raceway'],
                   'foot':['no'],
                   'service':['private'],
                   'access':['private']
                   }

highway_filters = filters['auto']['highway']

osm_highway_type_dict = {'motorway': ('motorway', False),
                         'motorway_link': ('motorway', True),
                         'trunk': ('trunk', False),
                         'trunk_link': ('trunk', True),
                         'primary': ('primary', False),
                         'primary_link': ('primary', True),
                         'secondary': ('secondary', False),
                         'secondary_link': ('secondary', True),
                         'tertiary': ('tertiary', False),
                         'tertiary_link': ('tertiary', True),
                         'residential': ('residential', False),
                         'residential_link': ('residential', True),
                         'service': ('service', False),
                         'services': ('service', False),
                         'cycleway': ('cycleway', False),
                         'footway': ('footway', False),
                         'pedestrian': ('footway', False),
                         'steps': ('footway', False),
                         'track': ('track', False),
                         'unclassified': ('unclassified', False)}

link_type_level_dict = {'motorway':1, 'trunk':2, 'primary':3, 'secondary':4, 'tertiary':5, 'residential':6, 'service':7,
                     'cycleway':8, 'footway':9, 'track':10, 'unclassified':11, 'connector':20, 'railway':30, 'aeroway':31}

default_lanes_dict = {'motorway': 4, 'trunk': 3, 'primary': 3, 'secondary': 2, 'tertiary': 2, 'residential': 1, 'service': 1,
                      'cycleway':1, 'footway':1, 'track':1, 'unclassified': 1, 'connector': 2}
default_speed_dict = {'motorway': 120, 'trunk': 100, 'primary': 80, 'secondary': 60, 'tertiary': 40, 'residential': 30, 'service': 30,
                      'cycleway':5, 'footway':5, 'track':30, 'unclassified': 30, 'connector':120}