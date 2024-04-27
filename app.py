from shiny import App, reactive, render, ui
from shinywidgets import render_widget, output_widget
from matplotlib import pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from urllib.request import urlopen
from scipy import stats
import pandas as pd
import json
import re

# declare variables
# colours
f1_red: str = '#E6002B'
main_bg_colour: str = '#111111'
text_colour: str = '#FFFFFF'
text_grey: str = '#AAAAAA'
default_driver0_colour: str = f1_red # '#1b14e3'
default_driver1_colour: str = text_colour
default_driver_colour: str = '#ae89c4'
track_border_colour: str = '#000000'
track_grey: str = '#001010'
# tracker_border_colour_driver1: str = text_colour

# lists for drop down selections
years: list[str] = ['2023','2024']
events: list[str] = []
sessions: list[str] = []
drivers: list[str] = []

# selected from drop down selections
selected_year: str = None
selected_meeting_key: str = None
selected_session_key: str = None
selected_driver_number: list[str] = []
selected_driver_colour: list[str] = []

# telemetry data downloaded retrieved
car_data: list[pd.DataFrame] = []
locations: list[pd.DataFrame] = []


# Shiny app layout
app_ui: ui.page_sidebar = ui.page_sidebar(
    ui.sidebar(
        ui.card(
            ui.input_selectize("year","Year: ", choices = years, selected = '2024'),
            ui.input_selectize("event","Event:", choices = [], selected = 0),
            ui.input_selectize("session","Session:", choices = []),
            ui.input_selectize('driver',"Driver:", choices = [], multiple = True),
            ui.input_action_button('refresh','Refresh', class_ = 'btn-success', style = 'color:' + text_colour  + '; background:' + f1_red + 'border: 1px solid ' + text_colour),
            height = 600,
            style = 'color:' + text_colour + '; background:' + f1_red,
        ),
        ui.output_text('ack'), width = '320px',
        bg = f1_red,
        fg = text_colour,
    ),
    ui.row(
        ui.column(3, ui.card(ui.output_plot("driver0"),height = 80, style = 'background:' + main_bg_colour)),
        ui.column(6, ui.output_text("header"),ui.output_text("sub_head"), height = 80, style = 'text-align:center; line-height:2; font-size:1.6em; font-weight:bold; color:' + text_colour + '; background:' + main_bg_colour + '; border: 1px solid ' + f1_red),
        ui.column(3, ui.card(ui.output_plot("driver1"),height = 80, style = 'background:' + main_bg_colour))
    ),
    ui.row(
        ui.column(
            3,
            ui.card(
                ui.card(ui.output_plot("driver0_throttle"), height = 100, style = 'background:' + main_bg_colour),
                ui.card(ui.output_plot("driver0_brake"), height = 100, style = 'background:' + main_bg_colour),
                ui.card(ui.output_plot("driver0_topspeed"), height = 100, style = 'background:' + main_bg_colour),
            height = 400,
            style = 'background:' + main_bg_colour,
            )
        ),
        ui.column(6, output_widget("tele_plot")),
        ui.column(
            3,
            ui.card(
                ui.card(ui.output_plot("driver1_throttle"), height = 100, style = 'background:' + main_bg_colour),
                ui.card(ui.output_plot("driver1_brake"), height = 100, style = 'background:' + main_bg_colour),
                ui.card(ui.output_plot("driver1_topspeed"), height = 100, style = 'background:' + main_bg_colour),
            height = 400,
            style = 'background:' + main_bg_colour
            )
        ),
    ),
    ui.row(
        ui.output_plot('track_dominance_plot'),
        height = 250,
    ),
    style = 'background:' + main_bg_colour
)

def server(input, output, session):
    def reset_variables(
        reset_events: bool = False,
        reset_sessions: bool = False,
        reset_drivers: bool = False,
        reset_selected_year:bool = False,
        reset_selected_meeting_key:bool = False,
        reset_selected_session_key: bool = False,
        reset_selected_driver_number: bool = False,
        reset_selected_driver_colour: bool = False,
        reset_car_data:bool = False,
        reset_locations:bool = False
    ):
        global events
        global sessions
        global drivers
        global selected_year
        global selected_meeting_key
        global selected_session_key
        global selected_driver_number
        global selected_driver_colour
        global car_data
        global locations
        
        events = [] if reset_events else events
        sessions = [] if reset_sessions else sessions
        drivers = [] if reset_drivers else drivers

        selected_year = None if reset_selected_year else selected_year
        selected_meeting_key = None if reset_selected_meeting_key else selected_meeting_key
        selected_session_key = None if reset_selected_session_key else selected_session_key
        selected_driver_number = [] if reset_selected_driver_number else selected_driver_number
        selected_driver_colour = [] if reset_selected_driver_colour else selected_driver_colour

        car_data = [] if reset_car_data else car_data
        locations = [] if reset_locations else locations

    def get_event_name(idx: int) -> str:
        return events.event_name[idx]

    def get_driver_name(idx: int) -> str:
        return drivers.full_name[idx]

    def get_driver_name_acronym(idx: int) -> str:
        return drivers.name_acronym[idx]

    def get_driver_num(idx: int) -> str:
        return str(drivers.driver_number[idx])

    def get_session_name(idx: int) -> str:
        return sessions.session_name[idx]

    def api_call(call_type: str, filters: list[str]) -> pd.DataFrame:
        call_str = 'https://api.openf1.org/v1/' + call_type + '?'

        for f in filters:
            if call_str[-1]  ==  '?':
                call_str  +=  f
            else:
                call_str  +=  '&' + f

        response = urlopen(call_str)
        df = pd.DataFrame(json.loads(response.read().decode('utf-8')))

        assert df.shape[0] > 0 and df.shape[1] > 0, "result pd.DataFrame is empty"        
        return df 

    def format_driver_names(driver_idx: int) -> str:
        names = get_driver_name(driver_idx).split()
        return '\n'.join([names[0],' '.join(names[1:])]), get_driver_name_acronym(driver_idx)

    def plot_names(front_name: str, back_name: str) -> plt.figure:
        fig, ax = plt.subplots()
        ax.text(0, 1, back_name, verticalalignment = 'top', horizontalalignment = 'left', color = text_grey, clip_on = False, fontstyle = 'oblique', fontweight = 'bold', fontfamily = 'sans-serif', fontsize = 36, alpha = 0.2)
        ax.text(1, 1, front_name, verticalalignment = 'top', horizontalalignment = 'right', color = text_colour, fontstyle = 'italic', fontfamily = 'sans-serif', fontsize = 13)
        fig.patch.set_facecolor(main_bg_colour)
        ax.set_facecolor(main_bg_colour)
        ax.axis('off')
        plt.tight_layout(pad = 0)
        return fig
    
    # number = index for driver in drivers
    def driver_metric(metric_name, idx, func, minval, maxval, display_name = None) -> plt.figure:
        fig, ax = plt.subplots()

        if len(input.driver()) > 0 and len(car_data) > idx:
            plot_data = car_data.copy()[idx]

            #print(plot_data[['throttle']].mean(), selected_driver_colour[num])
            metric = func(plot_data[[metric_name]]) #plot_data[[metric_name]].mean()
            ax.barh(0.1, metric, color = str(selected_driver_colour[idx]), height = 0.1, align = 'center')
            ax.text(maxval, 0.3, str(int(round(metric.iloc[0], 0))), verticalalignment = 'top', horizontalalignment = 'right', color = text_colour, fontstyle = 'oblique', fontweight = 'bold', fontfamily = 'sans-serif', fontsize = 22)
            display_name = metric_name if display_name is None else display_name
            ax.text(0, 0.3, display_name, verticalalignment = 'top', horizontalalignment = 'left', color = text_colour, fontstyle = 'oblique',fontweight = 'bold', fontfamily = 'sans-serif', fontsize = 14)

        ax.axis('off')
        ax.tick_params(axis = 'x', colors = text_colour, direction = 'in')
        ax.yaxis.set_visible(False)
        ax.set_xticks([])
        ax.set_xlim(minval, maxval)
        ax.set_ylim(0,0.3)
        ax.set_facecolor(main_bg_colour)
        fig.patch.set_facecolor(main_bg_colour)
        plt.tight_layout()

        return fig

    @reactive.effect
    def update_event_list_with_year() -> None:
        global events
        global selected_year

        # updatinig event drop-drop lists
        # reset drop-down lists
        ui.update_selectize("event",choices = [])
        ui.update_selectize("session",choices = [])
        ui.update_selectize("driver",choices = [])

        # reset variables for data and what are selected in drop-down lists
        reset_variables(reset_selected_year = None, reset_selected_meeting_key = None, reset_events = True, reset_sessions = True, reset_selected_session_key = True, reset_drivers = True, reset_selected_driver_number = True, reset_selected_driver_colour = True, reset_car_data = True, reset_locations = True )

        # get year from drop-down list
        selected_year = input.year()

        # get events data
        # update event drop-down list
        df = api_call('meetings', ['year=' + str(selected_year)])
        assert isinstance(df, pd.DataFrame)
        
        try:
            events = pd.concat([
                df.circuit_short_name,
                df.location,
                df.meeting_official_name.str.extract('FORMULA 1 (.*) 202.'),
                df.meeting_key,],axis = 1)\
                .rename(columns = {'circuit_short_name':'circuit_name','location':'location_name',0:'event_name'})

            ui.update_selectize("event",choices = events.event_name)
        except:
            print(f"Likely to be data error when retrieving event data from api_call: year={str(selected_year)}. Dataframe returned has {df.shape[0]} rows and {df.shape[1]} columns")

    @reactive.effect
    def update_session_list_with_event() -> None:
        global sessions
        global selected_meeting_key

        # updating session drop-down list
        # reset drop-down lists except for year and event
        ui.update_selectize("driver", choices = [])
        ui.update_selectize("session",choices = [])

        # reset variables
        reset_variables(reset_selected_meeting_key = True, reset_sessions = True, reset_selected_session_key = True, reset_drivers = True, reset_selected_driver_colour = True, reset_selected_driver_number = True, reset_car_data = True, reset_locations = True)

        # check drop-down lists year and event for values
        # get session data
        # update session drop-drop list
        if len(input.year()) > 0 and len(input.event()) > 0:
            selected_meeting_key = events.meeting_key[int(input.event())]

            df = api_call('sessions', ['year=' + str(selected_year),'meeting_key=' + str(selected_meeting_key)])
            assert isinstance(df, pd.DataFrame)

            try:
                sessions = pd.concat([
                    df.session_key,
                    df.date_start,
                    df.session_type,
                    df.session_name], axis = 1)

                ui.update_selectize("session",choices = sessions.session_name)
            except:
                print(f"Likely to be data error when retrieving session data from api_call: year={str(selected_year)}&meeting_key={str(selected_meeting_key)}. Dataframe returned has {df.shape[0]} rows and {df.shape[1]} columns")

    @reactive.effect
    def update_driver_list_with_session() -> None:
        global drivers
        global selected_session_key

        # updating driver drop-down list
        # reset drop-down lists except for year, event and session
        ui.update_selectize("driver", choices = [])

        # reset variables
        reset_variables(reset_drivers = True, reset_selected_driver_number = True, reset_selected_driver_colour = True, reset_car_data = True, reset_locations = True)

        if len(input.year()) > 0 and len(input.event()) and len(input.session()) > 0:
            selected_session_key = sessions.session_key[int(input.session())]

            df = api_call('drivers', ['session_key='+str(selected_session_key)])
            assert isinstance(df, pd.DataFrame)

            try:
                drivers = pd.concat([
                    df.driver_number,
                    df.name_acronym,
                    df.full_name,
                    df.team_name,
                    df.team_colour,
                    df.headshot_url,
                    df.country_code], axis = 1)
                drivers['team_name'] = drivers['team_name'].fillna('#NA')

                ui.update_selectize("driver", choices = drivers.full_name + " (" + drivers.team_name + ")")
            except:
                print(f"Likely to be data error when retrieving driver data from api_call: session={str(selected_session_key)}. Dataframe returned has {df.shape[0]} rows and {df.shape[1]} columns")

    # get car_data for fastest lap for selected driver
    @reactive.effect
    @reactive.event(input.refresh, ignore_none = False)
    def update_cardata_with_driver() -> None:
        global car_data
        global locations
        global selected_driver_number
        global selected_driver_colour

        # helper functions to map speed to car x, y position on track and determine track dominance at each trck position
        # get numerical attribute at specific time, e.g. speed
        def get_attr_from_car_data_datetime(car_data: pd.DataFrame, time: pd._libs.tslibs.timedeltas.Timedelta, attr:str) -> float:
            assert car_data[attr].dtypes in ['int64', 'float64']
            
            before = car_data.date - car_data.date.min() < time
            after = car_data.date - car_data.date.min() >=  time
            speed_before = -1 if len(car_data[before][attr])  ==  0 else car_data[before][attr].iloc[-1]
            speed_after = -1 if len(car_data[after][attr])  ==  0 else car_data[after][attr].iloc[0]
            return (speed_before + speed_after) / 2

        # Work out track dominance at specific position x, y between two cars
        # return 0 = first car is faster
        # return 1 = second car is faster
        # return -1 = same speed
        # x and y are car0 (driver0) position coordinates
        # compare with car1's (driver1's) speed at the closest x, y position
        def calc_dominance(x: int, y: int, speed: float) -> int:
            global locations
            loc1 = locations[1].copy()
            loc1['dist'] = abs(loc1.x - x) + abs(loc1.y - y)
            second_car_speed = loc1[loc1.dist  ==  loc1.dist.min()].speed.iloc[0]
            if speed > second_car_speed:
                return 0
            elif speed < second_car_speed:
                return 1
            else:
                return -1

        def get_fastest_lap(laps: pd.DataFrame) -> pd.DataFrame:
            # shortest lap time + has a sector 3 time (finished the lap)
            return laps[(laps.lap_duration  ==  laps.lap_duration.min()) & -laps.duration_sector_3.isna()]

        def get_lap_start_time(lap: pd.DataFrame) -> str:
            return lap.iloc[0].date_start

        def get_lap_end_time(lap:pd.DataFrame) -> str:
            lap_start = get_lap_start_time(lap)
            lap_duration = lap.iloc[0,:].lap_duration
            lap_end = str(pd.to_datetime(lap_start) + pd.Timedelta(seconds = lap_duration))
            return re.sub('\\s', 'T', lap_end)

        def map_car_data_attr_to_location_xy(location:pd.DataFrame, car_data:pd.DataFrame, attr:str) -> pd.DataFrame:
            location[attr] = -1.0
            for i in range(location.shape[0]):
                time = location.iloc[i,:].date
                location.loc[i,attr] = get_attr_from_car_data_datetime(car_data, time, 'speed')
            return location

        # smooth dominance data that are stored in locations[0] (driver0's location data
        def smooth_dominance(smooth: int) -> None:
            global locations
            for i in range(locations[0].shape[0]):
                if i + smooth < locations[0].shape[0]:
                    locations[0].loc[i, 'dominance'] = stats.mode(locations[0].loc[i:i + smooth, 'dominance'])[0]

        # Only update if 1 or 2 drivers are chosen
        if len(input.driver()) > 0 and len(input.driver()) <=  2:

            fastest_lap_start = None
            fastest_lap_end = None

            # store previously selected drivers
            old_selected_driver_number = selected_driver_number

            # update selected driver variables
            selected_driver_number = [drivers.driver_number[int(d)] for d in input.driver()]
            # assign team colour to driver or use default colour if team/team colour is none
            selected_driver_colour = ['#' + str(drivers.team_colour[int(d)]) if drivers.team_colour[int(d)] is not None else default_driver_colour for d in input.driver()]

            # if there are more than one driver and driver/team colour assigned are the same
            # assigned driver colours to be default driver0 and driver1 colours
            if len(input.driver()) > 1 and (selected_driver_colour[0]  ==  selected_driver_colour[1]):
                selected_driver_colour = [default_driver0_colour, default_driver1_colour]

            # get car_data and location_data
            temp_car_data = []
            temp_locations = []
            for i, d in enumerate(selected_driver_number):
                if old_selected_driver_number == [] or selected_driver_number[i] not in old_selected_driver_number:
                    api_result_df = api_call('laps',['session_key=' + str(selected_session_key),'driver_number=' + str(selected_driver_number[i])])

                    fastest_lap = get_fastest_lap(api_result_df)
                    fastest_lap_start = get_lap_start_time(fastest_lap)
                    fastest_lap_end = get_lap_end_time(fastest_lap)

                    car_data_df = api_call('car_data',['driver_number=' + str(selected_driver_number[i]),'session_key=' + str(selected_session_key),'date>=' + str(fastest_lap_start),'date<=' + str(fastest_lap_end)])
                    car_data_df['date'] = pd.to_datetime(car_data_df.date, format = 'mixed') - pd.to_datetime(car_data_df.date,format = 'mixed').min()
                    temp_car_data.append(car_data_df)

                    locations_df = api_call('location',['driver_number=' + str(selected_driver_number[i]),'session_key=' + str(selected_session_key),'date>=' + str(fastest_lap_start),'date<=' + str(fastest_lap_end)])
                    locations_df['date'] = pd.to_datetime(locations_df.date, format = 'mixed') - pd.to_datetime(locations_df.date,format = 'mixed').min()
                    start_x = locations_df[locations_df.date  ==  locations_df.date.min()].x.iloc[0]
                    start_y = locations_df[locations_df.date  ==  locations_df.date.min()].y.iloc[0]
                    locations_df['x'] = locations_df.x - start_x
                    locations_df['y'] = locations_df.y - start_y
                    temp_locations.append(locations_df)
                else:
                    temp_car_data.append(car_data.copy()[old_selected_driver_number.index(selected_driver_number[i])])
                    temp_locations.append(locations.copy()[old_selected_driver_number.index(selected_driver_number[i])])

            car_data = temp_car_data.copy()
            locations = temp_locations.copy()

            # initialize colour attribute in locations[0] for track dominance plotting
            # stored track dominance information only in locations[0] (driver0's location table) plotting
            locations[0]['colour'] = selected_driver_colour[0]

            # compare if there are two cars
            if len(locations) > 1:
                # map speed (car_data) to x, y position (location)
                for j in range(len(locations)):
                    locations[j] = map_car_data_attr_to_location_xy(locations[j],car_data[j], 'speed')
                    '''locations[j]['speed'] = -1.0
                    for i in range(locations[j].shape[0]):
                        time = locations[j].iloc[i,:].date
                        locations[j].loc[i,'speed'] = get_speed_from_car_data_datetime(car_data[j], time)'''

                # determine track dominance
                locations[0]['dominance'] = -99
                for i in range(locations[0].shape[0]):
                    x, y, speed = locations[0].iloc[i, :].x, locations[0].iloc[i, :].y, locations[0].iloc[i, :].speed
                    locations[0].loc[i, 'dominance'] = calc_dominance(x, y, speed)

                # smooth dominance result twice
                smooth_dominance(smooth = 12)
                smooth_dominance(smooth = 8)

                # assign driver colour to locations on track to represent track dominance
                for i in range(locations[0].shape[0]):
                    dominance = locations[0].loc[i, 'dominance']
                    if dominance  ==  0:
                        locations[0].loc[i, 'colour'] = selected_driver_colour[0]
                    elif dominance  ==  1:
                        locations[0].loc[i, 'colour'] = selected_driver_colour[1]
                    else:
                        locations[0].loc[i, 'colour'] = '#00AAAA'

        print(selected_meeting_key, selected_session_key)

    # header elements
    @render.text
    def header() -> str:
        if input.event():
            event_name = get_event_name(int(input.event()))
            return input.year() + ' ' + event_name if event_name !=  None else input.year()

    @render.text
    def sub_head() -> str:
        if input.session():
            session_name = get_session_name(int(input.session()))
            return session_name if session_name !=  None else ''

    @render.plot
    def driver0() -> plt.figure:
        if input.driver():
            front_name, back_name = format_driver_names(int(input.driver()[0]))
            return plot_names(front_name, back_name)

    @render.plot
    def driver1() -> plt.figure:
        if len(input.driver())  ==  2:
            front_name, back_name = format_driver_names(int(input.driver()[1]))
            return plot_names(front_name, back_name)

    # telemetry plot
    @render_widget
    @reactive.event(input.refresh, ignore_none = False)
    def tele_plot() -> go.Figure:
        fig = go.Figure()

        if input.driver() !=  '':
            plot_data = car_data.copy()

            for i, data in enumerate(plot_data):
                if i < 2:
                    data['date'] = data.date.astype(str)
                    data['date'] = data.date.str.extract(r'([0-9]{1}:[0-9]{2}.[0-9]{3})')
                    data.sort_values('date', inplace = True)
                    data['date'] = pd.to_datetime(data.date,format = '%M:%S.%f')
                    data.dropna(inplace = True)

                    dr_mask = drivers.driver_number  ==  selected_driver_number[i]

                    time_str = str(data.date.max() - data.date.min())
                    time_str = re.sub('[0-9]\s(days)\s','',time_str)
                    time_str = re.sub('(?<= .[0-9]{3})0{2,3}$','',time_str)
                    time_str = re.search('[0-9]{2}:[0-9]{2}.[0-9]{3}',time_str)
                    time_str = time_str.group() if time_str is not None else ''

                    fig.add_trace(go.Scatter(x = data.date, y = data.speed, line = (dict(color = str(selected_driver_colour[i]))) , name = list(drivers.loc[dr_mask, 'full_name'])[0] + ' ' + time_str))

        fig.update_layout(
            template = 'plotly_dark',
            plot_bgcolor = 'black',
            showlegend = True,
            legend = dict(yanchor = 'bottom',y = 0.01,xanchor = 'right',x = 0.99,orientation = 'h'))
        fig["layout"].update({"xaxis": {"tickformat": "%M:%S.%f"}})
        fig.update_yaxes(title_text = 'Speed / mph')
        return fig

    # driver metrics
    @render.plot
    @reactive.event(input.refresh, ignore_none = False)
    def driver0_throttle() -> plt.figure:
        return driver_metric('throttle', 0, pd.Series.mean, 0, 100, 'Throttle')

    @render.plot
    @reactive.event(input.refresh, ignore_none = False)
    def driver1_throttle() -> plt.figure:
        return driver_metric('throttle', 1, pd.Series.mean, 0, 100, 'Throttle')

    @render.plot
    @reactive.event(input.refresh, ignore_none = False)
    def driver0_brake() -> plt.figure:
        return driver_metric('brake', 0, pd.Series.mean, 0, 100, 'Brake')

    @render.plot
    @reactive.event(input.refresh, ignore_none = False)
    def driver1_brake() -> plt.figure:
        return driver_metric('brake', 1, pd.Series.mean, 0, 100, 'Brake')

    @render.plot
    @reactive.event(input.refresh, ignore_none = False)
    def driver0_topspeed() -> plt.figure:
        return driver_metric('speed', 0, pd.Series.max, 0, 350, 'Top speed')

    @render.plot
    @reactive.event(input.refresh, ignore_none = False)
    def driver1_topspeed() -> plt.figure:
        return driver_metric('speed', 1, pd.Series.max, 0, 350, 'Top speed')

    # track dominance
    @render.plot
    @reactive.event(input.refresh, ignore_none = False)
    def track_dominance_plot() -> go.Figure:
        global locations

        fig, ax = plt.subplots()

        if len(input.driver()) > 0 and len(locations) > 0:
            plot_data = locations.copy()[0]

            # draw driver0
            ax.plot(plot_data.x + 50, plot_data.y - 250, c = text_colour, ls = '-', lw = 10, alpha = 0.15)
            ax.plot(plot_data.x, plot_data.y, c = track_grey, ls = '-', lw = 6)
            ax.plot(plot_data.x, plot_data.y, c = selected_driver_colour[0], ls = '-', lw = 4)

            # work out segments for drawing driver1 dominant parts
            if len(plot_data.colour.unique()) > 1:
                current_segment = 0
                plot_data['segment'] = -1
                for i in range(plot_data.shape[0]):
                    plot_data.loc[i, 'segment'] = current_segment
                    if (i < plot_data.shape[0] - 1) and (plot_data.loc[i, 'dominance'] !=  plot_data.loc[i + 1, 'dominance']):
                        current_segment  +=  1

                # only draw driver1 dominant segments
                plot_data_c = plot_data[plot_data.dominance  ==  1]
                for seg in plot_data_c.segment.unique():
                        plot_segment = plot_data_c[plot_data_c.segment  ==  seg]
                        if plot_segment.shape[0] > 1:
                            # ax.plot(plot_segment.x, plot_segment.y, c = tracker_border_colour_driver1, ls = '-', lw = 6)
                            ax.plot(plot_segment.x, plot_segment.y, c = selected_driver_colour[1], ls = '-', lw = 4)

            ax.plot(plot_data.x[0], plot_data.y[0], c = track_border_colour, marker = 'o', markersize = 8)
            ax.plot(plot_data.x[0], plot_data.y[0], c = text_colour, marker = 'o', markersize = 6)

        ax.axis('off')
        ax.set_facecolor(main_bg_colour)

        fig.patch.set_facecolor(main_bg_colour)
        return fig
    
    @render.text
    def ack() -> str:
        return 'Credit to the open-sourced OpenF1.org API (https://openf1.org/) for the Formula One® telemetry data used in this dashboard'


app = App(app_ui, server)
