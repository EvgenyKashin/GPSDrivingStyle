# GPSDrivingStyle

This repo contains the Python project for parsing and analyzing GPS tracks in NMEA protocol. It also include a small dataset (18 rows) data.csv (processed features and labels) which I collected in two weeks driving my car. I labeled the data - agressive/no agressive style of driving. It may be used for predicting labels. Feel free to use it.

## Usage
Install the Python requirements with pip install -r requirements.txt. For comfort you can import script to Jupyter Notebook. For example:
```python
from gps_parser import *
%matplotlib inline
scatter_csv()
```
![scatter plot](/imgs/scatter.png)

### Data preparation
For collecting the data you may use your navigator or smartphone (my choose). If you use a smartphone application, check a GPS protocol of output - it should be NMEA.

Every single track should be placed in a separate folder called an integer number. For example '1' or '23' – name of the folder. It will be id of the track. Name the track 'data.txt' inside the folder. You should also create a description of the track in the same folder and call it 'desk.txt' - it may be useful when you will start analysis. The first word should be label, 'aggr' - aggressive style of driving, 'no' - no aggressive.

### Generating features
In the script gps_parser.py there are several primary function:
* process_data(id) - display features of the given track in a terminal or if you use Jupyter Notebook it also plot graphics
* add_to_csv(id) - add the row with features of the given track to the 'data.csv' file
* add_to_csv_range(range(10)) - will try to add first 10 tracks to 'data.csv'
* pairplot_csv() and scatter_csv() - draw graphics using the data from 'data.csv'

From terminal you can type:
```
python gps_parser.py id
```
where id is your integer id of the folder with data. This command will make process_data(id) and add_to_csv(id).

## TODO
Collect more data and build a predicting model.