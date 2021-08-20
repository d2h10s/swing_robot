from datetime import datetime
import time, os, yaml
from pytz import timezone, utc
from datetime import datetime as dt
dir = 'C:\\Users\\d2h10s\\OneDrive - 서울과학기술대학교\\repository\\paper\\a2c_serial\\logs\\Acrobot-v2_0819_15-38-25_test'
with open(os.path.join(dir, 'backup.yaml')) as f:
    yaml_data = yaml.safe_load(f)
    start_time_str = '2021_'+yaml_data['START_TIME']
start_time = datetime.strptime(start_time_str, '%Y_%m%d_%H-%M-%S')
print(type(start_time))
print(start_time_str)
print(utc.localize(dt.utcnow()).astimezone(timezone('Asia/Seoul')))
#time.mktime(t.timetuple()) + t.microsecond / 1E6
print(time.time())