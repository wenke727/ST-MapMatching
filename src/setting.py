DIS_FACTOR = 1/110/1000
DEBUG_FOLDER = "../debug"
GBA_BBOX = [112.471628,  22.138605, 114.424664,  23.565487]
SZ_BBOX  = [113.770651,  22.501331, 114.341732,  22.838288]


""" road_type_filter """
# Note: we adopt the filter logic from osmnx (https://github.com/gboeing/osmnx)
# exclude links with tag attributes in the filters
filters = {}

filters['auto'] = {'area':['yes'],
                   'highway':['cycleway','footway','path','pedestrian','steps','track','corridor','elevator','escalator',
                              'proposed','construction','bridleway','abandoned','platform','raceway','service'],
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
