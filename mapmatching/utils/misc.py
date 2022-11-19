from datetime import date

def get_date(fmt="%Y-%m-%d"):
    return date.today().strftime(fmt)

get_date()

