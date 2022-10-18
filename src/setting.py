""" Global config """
from pathlib import Path

root = Path(__file__).parent
DEBUG_FOLDER = root / "../debug"
CACHE_FOLDER = root / "../cache"
LOG_FOLDER   = root / "../log"
INPUT_FOLDER = root / "../input"

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